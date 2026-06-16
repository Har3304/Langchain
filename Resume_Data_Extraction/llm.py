import os
from google.colab import userdata
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from docling.document_converter import DocumentConverter

source = '' # Paste the path of resume document 

hf_token = userdata.get('HF_TOKEN_FINE')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

converter = DocumentConverter()
result= converter.convert(source)
text = result.document.export_to_markdown()

class ResumeExtract(BaseModel):
    name: str = Field(description="The name of the candidate.")
    email: str = Field(description="The email address of the candidate.")
    designation: str = Field(description = "The designation or professon of the candidate.")
    phone_number: str = Field(description="The phone number of the candidate.")
    education: str = Field(description="The educational qualifications of the candidate.")
    work_experience: dict = Field(description="The work experience of the candidate.")
    skills: dict = Field(description="The skills of the candidate.") 

parser = PydanticOutputParser(pydantic_object = ResumeExtract)
prompt = PromptTemplate(
    template="You are an expert data extraction assistant.\n{format_instructions}\nExtract data from this text strictly following the instructions: \n{raw_text}", 
    input_variables = ['raw_text'], 
    partial_variables = {'format_instructions':parser.get_format_instructions()}) 

llm = HuggingFaceEndpoint(repo_id = 'meta-llama/Llama-3.1-8B-Instruct', task = 'text-generation', temperature = 0.1)

chat_model = ChatHuggingFace(llm=llm)

extraction_chain = prompt | chat_model | parser


result = extraction_chain.invoke({'raw_text':text})
print(result)
