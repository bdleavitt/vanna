import json
import os
import pandas as pd
from typing import List
from ..base import VannaBase
from ..utils import deterministic_uuid
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    ComplexField,
    SearchField,
)

class AzureAISearch_VectorStore(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        if config is None:
            config = {}
        
        path = config.get("path", ".")
        self.embedding_function = config.get("embedding_function", self.get_aoai_embedding)
        curr_client = config.get("client", "default")
        collection_metadata = config.get("collection_metadata", None)
        self.index_prefix = config.get("index_prefix", "vanna-")
        self.n_results_sql = config.get("n_results_sql", config.get("n_results", 10))
        self.n_results_documentation = config.get("n_results_documentation", config.get("n_results", 10))
        self.n_results_ddl = config.get("n_results_ddl", config.get("n_results", 10))

        # Define the Search Client
        if isinstance(curr_client, SearchIndexClient):
            self.aisearch_client = curr_client
        elif curr_client == "default":
            self.aisearch_client = SearchIndexClient(
                endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                credential=AzureKeyCredential(key=os.getenv("AZURE_AI_SEARCH_ADMIN_KEY")),
            )
        else:
            raise ValueError(f"Unsupported client was set in config: {curr_client}")
        print("Initialized client to Azure AI Search.")

        # Define the AOAI Client
        self.aoai_client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version= os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            api_key = os.getenv("AZURE_OPENAI_KEY")
        )
        print("Initialized client to Azure OpenAI.")
        
        ## Documentation Collection Index
        self.documentation_index_client = self.create_collection_index("documentation")

        ## DDL Collection Index
        self.ddl_index_client = self.create_collection_index("ddl")
        
        ## SQL COLLECTION
        self.sql_index_client = self.create_collection_index("sql")

    def get_aoai_embedding(self, text:str, embedding_model_deployment:str=None):
        # use the model specificed in environment variables if not passed as an env
        if embedding_model_deployment is None:
            embedding_model_deployment =  os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        else: 
            embedding_model_deployment = embedding_model_deployment
    
        response = self.aoai_client.embeddings.create(
            input = [text], 
            model=embedding_model_deployment
        )
        embeddings = response.data[0].embedding
        return embeddings

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function(data)
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:      
        question_sql_dict = {
            "question": question,
            "sql": sql,
        }
        question_sql_json = json.dumps(
            question_sql_dict,
            ensure_ascii=False,
        )
        id = deterministic_uuid(question_sql_json) + "-sql"
        embedding = self.generate_embedding(question_sql_json)
        res = self.sql_index_client.merge_or_upload_documents(
            {
                "id": id,
                "content": question_sql_dict,
                "vector": embedding
            }
        )
        
        return id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = deterministic_uuid(ddl) + "-ddl"
        self.ddl_index_client.merge_or_upload_documents(
            {
                "id": id,
                "content": ddl,
                "vector": self.generate_embedding(ddl)
            }
        )
        return id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        id = deterministic_uuid(documentation) + "-doc"
        self.documentation_index_client.merge_or_upload_documents(
            {
                "id": id,
                "content": documentation,
                "vector": self.generate_embedding(documentation)
            }
        )
        return id
        
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_index_client.search(search_text="*", select="id, content")
        sql_data = [doc for doc in sql_data]

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [doc['content'] for doc in sql_data]
            ids = [doc['id'] for doc in sql_data]

            # Create a DataFrame
            df_sql = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_index_client.search(search_text="*", select="id, content")
        ddl_data = [doc for doc in ddl_data]

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [doc['content'] for doc in ddl_data]
            ids = [doc['id'] for doc in ddl_data]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_index_client.search(search_text="*", select="id, content")
        doc_data = [doc for doc in doc_data]
        if doc_data is not None:
            # Extract the documents and ids
            documents = [doc['content'] for doc in doc_data]
            ids = [doc['id'] for doc in doc_data]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            self.sql_index_client.delete_documents([{ "id": id }])
            return True
        elif id.endswith("-ddl"):
            self.ddl_index_client.delete_documents([{ "id": id }])
            return True
        elif id.endswith("-doc"):
            self.documentation_index_client.delete_documents([{ "id": id }])
            return True
        else:
            return False

    def remove_collection(self, collection_name: str) -> bool:
        """
        This function can reset the collection to empty state.

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        if collection_name == "sql" or collection_name == self.sql_index_client._index_name:
            self.aisearch_client.delete_index(index=self.sql_index_client._index_name)
            self.sql_index_client = self.create_collection_index(
                collection_name="sql"
            )
            return True
        elif collection_name == "ddl" or collection_name == self.ddl_index_client._index_name:
            self.aisearch_client.delete_index(index=self.ddl_index_client._index_name)
            self.ddl_index_client = self.create_collection_index(
                collection_name="ddl"
            )
            return True
        elif collection_name == "documentation" or collection_name == self.documentation_index_client._index_name:
            self.aisearch_client.delete_index(index=self.documentation_index_client._index_name)
            self.documentation_index_client = self.create_collection_index(
                collection_name="documentation"
            )
            return True
        else:
            return False

    @staticmethod
    def _extract_documents(query_results) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.

        Returns:
            List[str] or None: The extracted documents, or an empty list or
            single document if an error occurred.
        """
        print(query_results)

        if query_results is None:
            return []

        documents = [doc for doc in query_results]

        ## TODO: investigate what this logic is for
        if len(documents) == 1 and isinstance(documents[0], list):
            try:
                documents = [json.loads(doc) for doc in documents[0]]
            except Exception as e:
                return documents[0]

        return documents

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        ## TODO: update to allow hybrid search and other vector parameters
        embedded_question = self.generate_embedding(question)
        
        vector_query = VectorizedQuery(
            vector=embedded_question,
            k_nearest_neighbors=self.n_results_sql, ## todo -- does this parameter makes sense?
            fields="vector"
        )
        
        results = self.sql_index_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id, content"]
        )

        documents = [doc for doc in results]

        return AzureAISearch_VectorStore._extract_documents(
            query_results=documents
        )

    def get_related_ddl(self, question: str, **kwargs) -> list:      
        embedded_question = self.generate_embedding(question)
        
        vector_query = VectorizedQuery(
            vector=embedded_question,
            k_nearest_neighbors=self.n_results_ddl, ## todo -- does this parameter makes sense?
            fields="vector"
        )
        
        results = self.ddl_index_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id, content"]
        )

        documents = [doc for doc in results]

        return AzureAISearch_VectorStore._extract_documents(
            query_results=documents
        )

    def get_related_documentation(self, question: str, **kwargs) -> list:
        embedded_question = self.generate_embedding(question)
        
        vector_query = VectorizedQuery(
            vector=embedded_question,
            k_nearest_neighbors=self.n_results_documentation, ## todo -- does this parameter makes sense?
            fields="vector"
        )
        
        results = self.documentation_index_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id, content"]
        )

        documents = [doc for doc in results]

        return AzureAISearch_VectorStore._extract_documents(
            query_results=documents
        )
    
    def create_collection_index(self, collection_name: str) -> SearchClient:
        ## vector search config
        algorithm_config_name = "hnsw_vanna_algo_config"
        vector_search_profile_name = "hnsw_vanna_vector_profile"

        vector_search_config = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=algorithm_config_name
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name=vector_search_profile_name,
                    algorithm_configuration_name=algorithm_config_name,
                )
            ]
        )

        if collection_name.lower() == 'sql':
            fields =[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                # SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
                ComplexField(name="content", collection=False, fields=[
                    SearchableField(name="question", type=SearchFieldDataType.String, searchable=True),
                    SearchableField(name="sql", type=SearchFieldDataType.String, searchable=True)
                ]),
                SearchField(name="vector", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, 
                            vector_search_dimensions=1536,
                            vector_search_profile_name="hnsw_vanna_vector_profile"
                )
            ]
        elif collection_name.lower() == 'ddl':
            fields =[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="vector", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, 
                            vector_search_dimensions=1536,
                            vector_search_profile_name="hnsw_vanna_vector_profile"
                )
            ]
        elif collection_name.lower() == 'documentation':
            fields =[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="vector", 
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, 
                            vector_search_dimensions=1536,
                            vector_search_profile_name="hnsw_vanna_vector_profile"
                )
            ]
        else:
            raise ValueError(f"Unsupported collection name: {collection_name}") 
        
        index_name = self.index_prefix + collection_name
        
        print(f"Creating index and client for {collection_name} collection. Index: {index_name}")    
        
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search_config)
        
        self.aisearch_client.create_or_update_index(index)

        return self.aisearch_client.get_search_client(index_name)