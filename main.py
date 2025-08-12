from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.agents import initialize_agent, AgentType
from langchain.prompts.chat import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
)
from defs_req import load_model, recovery_chunks, rag_extration_requisitos, recovery_chunks, request_exatraction, prompt_rag_structured, banco_doc_licitacao, create_vectorstore_req
from defs_analist import prompt_analista, query_analise 
from tools import tool_rag, tool_alimentar_requisitos, adicionar_requisito_tool, tool_verificar_progresso
import os


#======== EXTRAÇÃO DE REQUISITOS ========##s

# cria o vectorstore do banco de documentos de licitação 
vectorstore_licitacao = banco_doc_licitacao()

# retorna as partes do documento em texto dos requisitos
partes = recovery_chunks(vectorstore_licitacao)
  
# carrega o modelo
llm = load_model("gpt-4.1-mini")

# carrega o prompt estruturados
prompt = prompt_rag_structured()

# faz a chain do prompt e do modelo
chain = prompt | llm

# faz a query para a extracao
pergunta = request_exatraction()

# faz a extracao de requisitos e retorna os resultados no arquivo csv
rag_extration_requisitos(chain=chain, pergunta=pergunta, partes=partes)

# cria a vectorstore para os requisitos do arquivo csv
vectorstore_req = create_vectorstore_req()

##======== TREINAMENTO DO MODELO ========##

system_message_prompt = prompt_analista()
query = query_analise()


template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message_prompt),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

agent = initialize_agent(
    tools=[tool_rag, tool_alimentar_requisitos, adicionar_requisito_tool, tool_verificar_progresso],
    llm=llm,
    temperature=0.1,
    max_retries=0,
    retry_min_seconds=1, 
    retry_max_seconds=60,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prompt=template,
    max_iterations=100000,
    early_stopping_method="force",
    handle_parsing_errors=True,
    max_execution_time=10000000000,
    verbose=True,
)
result = agent.invoke({"input": query})
print("Resposta final:", result.get("output", "Sem resposta"))
print("Passos intermediários:", result.get("intermediate_steps", "Sem passos intermediários"))
