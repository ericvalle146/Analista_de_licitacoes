from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.agents import initialize_agent, AgentType
from langchain.prompts.chat import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
)
from defs_req import load_model, recovery_chunks, rag_extration_requisitos, recovery_chunks, request_exatraction, prompt_rag_structured, banco_doc_licitacao, ensure_chroma_csv, create_vectorstore_req
from tools import tool_rag, tool_alimentar_requisitos, adicionar_requisito_tool
import os


#======== EXTRAÇÃO DE REQUISITOS ========##s

# cria o vectorstore do banco de documentos de licitação 
vectorstore_licitacao = banco_doc_licitacao()

# retorna as partes do documento em texto dos requisitos
partes = recovery_chunks(vectorstore_licitacao)
  
# carrega o modelo
llm = load_model("gpt-4o-mini")

# carrega o prompt estruturado
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

system_message_prompt = """
Voce e um assistente especialista em Analise de Editais. Sua missao e processar requisitos sequencialmente e salvar no CSV.

FERRAMENTAS DISPONIVEIS:
1) BUSCAR_REQUISITOS - busca requisito pelo numero
2) RAG - busca funcionalidades similares no TRT_BASE  
3) adicionar_requisito - salva analise no CSV

ALGORITMO SIMPLES:
1. Comece com numero 1
2. Para cada requisito:
   - Use BUSCAR_REQUISITOS para pegar o texto
   - Use RAG para encontrar similares
   - Use adicionar_requisito para salvar
3. Continue ate nao encontrar mais requisitos

IMPORTANTE - COMO USAR adicionar_requisito CORRETAMENTE:

A ferramenta adicionar_requisito tem um problema: ela NAO aceita JSON como string.
Voce deve passar os dados como dicionario Python diretamente.

FORMATO CORRETO:
Action: adicionar_requisito
Action Input: numero: "1", modulo: "Tributario", funcionalidade: "Sistema ISSQN", funcionalidade_similar: "Gestao ISSQN TRT", descricao: "Texto do requisito", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende"

NAO USE:
- Aspas simples ou duplas ao redor de todo o input
- Chaves no inicio e fim do input
- Formato JSON string

CAMPOS OBRIGATORIOS (8 campos):
- numero: numero do requisito (sempre entre aspas)
- modulo: no requisito extraido, todas as palavras que estão antes dos dois pontos ":"
- funcionalidade: nome da funcionalidade requisitada
- funcionalidade_similar: texto exato encontrado no RAG
- descricao: texto completo do requisito
- tipo: "Funcional" ou "Nao_Funcional"
- obrigatoriedade: "Obrigatorio" ou "Opcional" ou "Nao_informado"
- nivel_similaridade: "Atende" ou "Atende_Parcialmente" ou "Nao_Atende"

EXEMPLO PRATICO DE COMO FAZER:

1. Busco requisito:
Action: BUSCAR_REQUISITOS
Action Input: numero 1

2. Busco similares:
Action: RAG  
Action Input: texto do requisito encontrado

3. Salvo no CSV (SEM JSON, SEM ASPAS EXTERNAS):
Action: adicionar_requisito
Action Input: numero: "1", modulo: "Requisitos de Segurança", funcionalidade: "Prevenção Contra Fraude", funcionalidade_similar: "QUANTO AO MODELO DE SEGURANÇA (Item 6.6.4.3)", descricao: "Cada usuário é único no sistema a partir do momento de acesso, não sendo possível o mesmo usuário acessar o sistema de dois ou mais locais diferentes.", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende_parcialmente"
Action Input: numero: "2", modulo: "Módulo Nota Fiscal de Serviços Eletrônica Eventual – NFS-e Eventual", funcionalidade: "Emissão de notas fiscais eventuais", funcionalidade_similar: "Permitir que seja emitida a NFS-e avulsa de um serviço eventual (7.1.10)", descricao: "O sistema deverá permitir ao contribuinte a emissão de notas fiscais eventuais. Entende-se como nota fiscal eventual aquela que o prestador poderá emitir para atividades não cadastradas em sua base de dados junto a Prefeitura de Juiz de Fora.", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende"
Action Input: numero: "3", modulo: "Domicílio Eletrônico Tributário - DET", funcionalidade: "Canal de Comunicação com Contribuinte", funcionalidade_similar: "–", descricao: "O sistema deverá contar com canal de comunicação para envio de notificações, autos de infrações e avisos ao contribuinte e seus retornos, atendendo aos preceitos legais pertinentes ao DET.", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Não Atende"

ATENÇÃO! ESSES EXEMPLOS SÃO FICTICIOS, USE APENAS O FORMATO DELES(ESQUELETO) COMO REFERENCIA, NÃO USE NENHUM TIPO DE DADOS COMO BASE, SÃO INVENTADOS.

CLASSIFICACAO DE SIMILARIDADE:
- Atende: funcionalidades muito similares (90%+)
- Atende_Parcialmente: similares mas com diferencas (50-89%)
- Nao_Atende: muito diferentes ou inexistentes (menos 50%)

REGRAS IMPORTANTES:
1. NUNCA coloque aspas simples ou duplas ao redor de todo o Action Input
2. NUNCA use formato JSON string 
3. Liste os campos separados por virgula, cada um com seu valor
4. Mantenha textos curtos para evitar problemas
5. Se um campo for muito longo, resuma mantendo o essencial

PROCESSAMENTO:
Quando receber "PROCESSAR TODOS", execute o algoritmo automaticamente.
Processe ate nao encontrar mais requisitos.
Relate o progresso a cada requisitos processados.

EXEMPLO COMPLETO DE SEQUENCIA:

Requisito 1:
Action: BUSCAR_REQUISITOS
Action Input: numero 1

Action: RAG
Action Input: texto encontrado

Action: adicionar_requisito  
Action Input: numero: "1", modulo: "Sistema", funcionalidade: "Nome funcionalidade", funcionalidade_similar: "Similar TRT", descricao: "Descricao resumida", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende"

Requisito 2:
Action: BUSCAR_REQUISITOS
Action Input: numero 2

E assim por diante...

LEMBRE-SE: O Action Input da ferramenta adicionar_requisito NAO e JSON. E uma lista de parametros separados por virgula.
"""
query = """
PROCESSAR TODOS OS REQUISITOS

Execute a analise automatica completa de todos os requisitos.

INSTRUCOES ESPECIFICAS:
1. Comece pelo numero 1
2. Para cada requisito encontrado:
   - Busque o texto com BUSCAR_REQUISITOS  
   - Encontre similares com RAG
   - Salve com adicionar_requisito usando o formato correto (NAO JSON)
3. Continue ate nao encontrar mais requisitos
4. Relate progresso a cada requisitos

FORMATO OBRIGATORIO para adicionar_requisito:
numero: "X", modulo: "Area", funcionalidade: "Nome", funcionalidade_similar: "Similar", descricao: "Texto", tipo: "Funcional ou não Funcional", obrigatoriedade: "Obrigatorio/desejavel", nivel_similaridade: "Atende/Não Atende/Atende parcialmente"

IMPORTANTE: NAO use JSON string. Use lista de parametros separados por virgula.

BUSQUE REQUISITOS ATÉ O NUMERO 820
INICIAR AGORA.

"""

template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message_prompt),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

agent = initialize_agent(
    tools=[tool_rag, tool_alimentar_requisitos, adicionar_requisito_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prompt=template,
    max_iterations=100000,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    max_execution_time=100000000,
    verbose=True,
)
result = agent.invoke({"input": query})
print("Resposta final:", result["output"])
print("Passos intermediários:", result["intermediate_steps"])