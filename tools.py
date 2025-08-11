from langchain.agents import Tool
from defs_train import rag_banco_base, adicionar_requisito
from defs_req import alimentacao_req

tool_rag = Tool(
    name="RAG",
    func=rag_banco_base,
    description="Busca os 3 documentos mais similares no banco vetorial com base na query.",
)

tool_alimentar_requisitos = Tool(
    name="BUSCAR_REQUISITOS",
    func=alimentacao_req,
    description="Buscar o requisito no banco vetorial com base no numero.",
)

adicionar_requisito_tool = Tool(
    name="adicionar_requisito",
    func=adicionar_requisito,
    description="""
    Adiciona requisito ao arquivo CSV.
    
    Use exatamente este formato:
    numero="X", modulo="todas as palavras que estiverem antes dos dois pontos ":" contino no requisito", funcionalidade="Nome", funcionalidade_similar="Similar", descricao="Texto", tipo="Funcional", obrigatoriedade="Obrigatorio", nivel_similaridade="Atende"
    
    Parametros obrigatorios (todos entre aspas duplas):
    - numero: numero do requisito
    - modulo: todas as palavras que estiverem antes dos dois pontos ":" contino no requisito
    - funcionalidade: nome da funcionalidade  
    - funcionalidade_similar: funcionalidade do TRT_BASE
    - descricao: texto do requisito
    - tipo: Funcional ou Nao_Funcional
    - obrigatoriedade: Obrigatorio ou Opcional
    - nivel_similaridade: Atende ou Atende_Parcialmente ou Nao_Atende
    """
)
