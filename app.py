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

app = VannaFlaskApp(vn, title="Vanna.AI - With Azure AI Search and Azure OpenAI Chat")
app.run()