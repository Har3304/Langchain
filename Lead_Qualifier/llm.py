import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from google.colab import userdata


hf_token = userdata.get('HF_TOKEN_FINE')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token


class CompanyLead(BaseModel):
    company_name: str = Field(description="The formal name of the company.")
    industry: str = Field(description="The primary sector or vertical.")
    is_qualified: bool = Field(description="True if the company offers B2B services.")

parser = PydanticOutputParser(pydantic_object=CompanyLead)
prompt = PromptTemplate(template="You are an expert data extraction assistant.\n{format_instructions}\nExtract data from this text:\n{raw_text}",
    input_variables=["raw_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()})

llm = HuggingFaceEndpoint(
    task="text-generation",
    temperature=0.1)

chat_model = ChatHuggingFace(llm=llm)
extraction_chain = prompt | chat_model | parser
sample_text = "Apex Logistics Solutions operates in the freight sector and provides B2B contracts."
result = extraction_chain.invoke({"raw_text": sample_text})
print(result)
