from concurrent.futures import as_completed
import re
from langchain_core.runnables.config import ContextThreadPoolExecutor
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict

from knowledge_db import DB
from doc_db import DocumentationDB
from utils.logger_manager import CustomLogger
from utils.utilities import extract_json, read_file, create_output_dict, extract_list
from utils.constants import (
    BEST_PRACTICES_CHUNK_SIZE,
    MAX_FILES_PROCESS_CONCURRENTLY,
    MAX_CHUNKS_PROCESS_CONCURRENTLY,
    LOG_FILE_PATH,
    FILE_THREAD_PREFIX,
    CHUNK_THREAD_PREFIX,
    STATEMENT_KEYWORD_PATTERN,
    CODE_LINES_CHUNK_SIZE,
    DOCUMENTATION_SPLIT_MODEL,
    DOCUMENTATION_CHUNK_OVERLAP,
    DOCUMENTATION_CHUNK_TOKEN_LIMIT,
    CODE_LEVEL_PROMPT,
    UNKNOWN_RESPONSE_PROMPT,
    SYSTEM_PROMPT,
    MODEL_NAME,
    EMBEDING_MODEL,
    AI71_BASE_URL
)

class CodeAnalyzer:

    def __init__(
        self, project_frameworks, file_practice_mapping, processed_best_practice_dict_list
    ):
        self.project_frameworks = project_frameworks
        self.file_practice_mapping = file_practice_mapping
        self.best_practice_keyword_dict = processed_best_practice_dict_list
        self.db_store = DB()
        self.logger = CustomLogger(LOG_FILE_PATH).get_logger()

    def split_file_by_code_lines_limit(self, input_file_path: str, allowed_lines_per_file: int):
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        non_empty_lines = [line for line in lines if line.strip()]
        contents = []
        for i in range(0, len(non_empty_lines), allowed_lines_per_file):
            contents.append(non_empty_lines[i : i + allowed_lines_per_file])
        return contents

    def combine_chunk_responses(self, curr_response, processed_response):
        for kw in processed_response:
            if kw in curr_response:
                curr_response[STATEMENT_KEYWORD_PATTERN].extend(processed_response[kw])
            else:
                curr_response[kw] = processed_response[kw]

    def query_existing_data_in_knowledge_store(self, best_practices_list, file_path_str, old_responses):
        code = read_file(file_path_str)
        found_best_practice = []
        for best_practice in best_practices_list:
            try:
                found = re.search(STATEMENT_KEYWORD_PATTERN, best_practice, re.DOTALL)
                if found:
                    statement, keyword = found.groups()
                    statement = statement.strip()
                    keyword = keyword.strip()
                    res = self.db_store.query_file(code, statement)

                    if res:
                        if file_path_str not in old_responses:
                            old_responses[file_path_str] = {}
                        old_responses[file_path_str][keyword] = res
                        found_best_practice.append(best_practice)
                        self.logger.info(
                            f"Found exisitng response for {best_practice} in knowledge store"
                        )
            except Exception as e:
                self.logger.error(f"Error in querying knowledge store or response is not found in knowledge store: {str(e)}")

        remaining_best_practice = []
        for i in range(len(best_practices_list)):
            best_practice = best_practices_list[i]
            if best_practice not in found_best_practice:
                remaining_best_practice.append(best_practice)

        return remaining_best_practice

    def combine_responses_into_final_response(self, old_res, new_res):
        try:
            for file_path in new_res:
                if file_path not in old_res:
                    old_res[file_path] = {}
                old_res[file_path].update(new_res[file_path])

            final_response = {}
            for file_path in old_res:
                final_response[file_path] = {}
                for keyword in old_res[file_path]:
                    if not len(old_res[file_path][keyword]):
                        continue
                    all_violated_response = [
                        response
                        for response in old_res[file_path][keyword]
                        if response["status"].lower() == "violated"
                    ]
                    if len(all_violated_response):
                        final_response[file_path][keyword] = all_violated_response

            return final_response
        except Exception as e:
            self.logger.error(f"Error while combing old and new response: {str(e)}")
            return {}

    def execute_file_analyze(self):
        old_responses = {}
        final_response={}
        try:
            with ContextThreadPoolExecutor(
                max_workers=MAX_FILES_PROCESS_CONCURRENTLY,
                thread_name_prefix=FILE_THREAD_PREFIX,
            ) as executor:
                future_to_file = {}
                for file_path, best_practices in self.file_practice_mapping.items():
                    remaining_best_practices = self.query_existing_data_in_knowledge_store(
                        best_practices, file_path, old_responses
                    )
                    if not len(remaining_best_practices):
                        continue

                    all_chunks = self.split_file_by_code_lines_limit(file_path, CODE_LINES_CHUNK_SIZE)
                    for each_file_chunk in all_chunks:
                        future_to_file[
                            executor.submit(
                                self.analyze_file_chunk_concurrently,
                                each_file_chunk,
                                file_path,
                                remaining_best_practices,
                            )
                        ] = file_path
                unknown_handled_practices = []
                new_responses = {}
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        response = future.result()
                        if file_path not in new_responses:
                            new_responses[file_path] = {}
                        self.combine_chunk_responses(
                            new_responses[file_path], self._handle_unknown_responses(
                                response, file_path, unknown_handled_practices
                            )
                        )
                    except Exception as exc:
                        self.logger.error(
                            f"File analysis failed for {file_path}: {exc}"
                        )
            for file_path in new_responses:
                for best_practice_keyword in new_responses[file_path]:
                    try:
                        best_practice_statement = self.best_practice_keyword_dict[
                            best_practice_keyword
                        ]["statement"]
                        file_content = read_file(file_path)
                        self.db_store.insert_file(
                            file_content,
                            best_practice_statement,
                            new_responses[file_path][best_practice_keyword],
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error while storing {file_path} with best practice {best_practice_statement} in db store: {str(e)}"
                        )
            final_response = self.combine_responses_into_final_response(old_responses, new_responses)
        except Exception as e:
            self.logger.error(f"Threading failed: {e}")

        self.logger.info("Code level analysis completed.")
        return final_response

    def analyze_file_chunk_concurrently(self, file_chunk_str, file_path_str, best_practices_list):
        combined_data = {}
        try:
            with ContextThreadPoolExecutor(
                max_workers=MAX_CHUNKS_PROCESS_CONCURRENTLY,
                thread_name_prefix=CHUNK_THREAD_PREFIX,
            ) as executor:
                futures = [
                    executor.submit(self.analyze_chunk, file_chunk_str, file_path_str, chunk)
                    for chunk in self._chunk_best_practices(best_practices_list)
                ]
                for future in as_completed(futures):
                    try:
                        chunk_result = future.result()
                        combined_data.update(chunk_result)
                    except Exception as exc:
                        self.logger.error(
                            f"Chunk analysis failed for {file_path_str}: {exc}"
                        )
        except Exception as e:
            self.logger.error(f"Chunk thread pool setup failed for {file_path_str}: {e}")
        return combined_data
