import os

from dotenv import load_dotenv

from vanna.openai import OpenAI_Chat
from vanna.azureaisearch import AzureAISearch_VectorStore
from vanna.flask import VannaFlaskApp

load_dotenv()

class AzureVanna(AzureAISearch_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        AzureAISearch_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=self.aoai_client, config=config) # Make sure to put your AzureOpenAI client here

vn = AzureVanna(config={"model": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")})

server = os.getenv('AZURE_SQL_DB_SERVER')
database = os.getenv('AZURE_SQL_DB_NAME')
username = os.getenv('AZURE_SQL_DB_USER')
password = os.getenv('AZURE_SQL_DB_PASSWORD')

conn_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

vn.connect_to_mssql(odbc_conn_str=conn_string) # You can use the ODBC connection string here

app = VannaFlaskApp(vn,  title="Vanna.AI - With Azure AI Search and Azure OpenAI Chat")
app.run()