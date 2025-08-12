from langchain.agents import Tool
from defs_analist import rag_banco_base, adicionar_requisito, verificar_ultimas_linhas
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
    
    FORMATO OBRIGATÓRIO (todos os 8 campos):
    numero="X", modulo="X", funcionalidade="X", funcionalidade_similar="X", descricao="X", tipo="Funcional", obrigatoriedade="Obrigatorio", nivel_similaridade="Atende"
    
    CAMPO NIVEL_SIMILARIDADE É OBRIGATÓRIO:
    - Use EXATAMENTE: "Atende", "Atende_parcialmente" ou "Nao_atende"  
    - NUNCA deixe vazio!
    
    Todos os parâmetros devem estar entre aspas duplas.
    """
)

tool_verificar_progresso = Tool(
    name="Verificar_Progresso",
    func=verificar_ultimas_linhas,
    description="Verifica as duas últimas linhas do arquivo analise.csv para ver o progresso. Útil para continuar de onde parou."
)