from langchain.agents import Tool
from defs_analist import rag_banco_base, adicionar_requisito
from defs_req import alimentacao_req

tool_rag = Tool(
    name="RAG",
    func=rag_banco_base,
    description="Busca uma função similar no banco vetorial com base na query.",
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
    
    Use exatamente este modelo de esqueleto(exemplo apenas):
    numero="X", modulo="todas as palavras que estiverem antes dos dois pontos ":" contido no requisito", funcionalidade="tudo depois dos dois primeiros dois prontos ":", contido no requisito ", funcionalidade_similar="Fucionalidade similar encontrada com a função RAG", descricao="descrição detalhado do requisito", tipo="Funcional", obrigatoriedade="Obrigatorio", nivel_similaridade="Atende/Atende parcialmente/Não atende"
    
    Parametros obrigatorios (todos entre aspas duplas):
    - numero: numero do requisito
    - modulo: todas as palavras que estiverem antes dos dois pontos ":" contino no requisito
    - funcionalidade: nome da funcionalidade  
    - funcionalidade_similar: funcionalidade do TRT_BASE
    - descricao: descrição detalhado do requisito
    - tipo: Funcional ou Nao Funcional
    - obrigatoriedade: Obrigatorio ou Opcional
    - nivel_similaridade: Atende ou Atende Parcialmente ou Nao Atende
    """
)
