from concurrent.futures import as_completed
import re
from langchain_core.runnables.config import ContextThreadPoolExecutor
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain


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
        a=combine_chunk_responses(file_path_str,4)
        for best_practice in best_practices_list:
            try:
                found = re.search(STATEMENT_KEYWORD_PATTERN, best_practice, re.DOTALL)
                if found:
                    statement, keyword = found.groups()
                    statement = statement.strip()
                    keyword = keyword.strip()
                    print(keyword)
                    res = self.db_store.query_file(code, statement)
                    print(res)
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
