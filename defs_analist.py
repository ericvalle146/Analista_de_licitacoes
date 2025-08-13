from vectorstore import ensure_chroma
from typing import Dict
import os, json , csv, re
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


#====== ANALISE ENTRE DOCUMENTOS ======##

# prompt da analise
def prompt_analista():
    system_message_prompt = """
VOCÊ É UM ASSISTENTE ESPECIALISTA EM ANÁLISE DE EDITAIS

OBJETIVO: Para cada requisito, gerar análise crítica e registrar com adicionar_requisito no formato obrigatório.

📌 PASSOS (por requisito):
1) BUSCAR_REQUISITOS(numero)
2) Extrair MÓDULO = texto antes dos dois pontos ':' no requisito (ex.: "Requisito Funcional: ..." → "Requisito Funcional")
3) Extrair FUNCIONALIDADE = texto após os dois pontos
4) Criar 2 QUERIES RAG distintas:
   - Query A: escolher 3–4 palavras-chave compostas (bigram/trigram) altamente específicas do requisito.
       Ex.: "confirmação chegada paciente", "registro ausência consulta", "autenticação biometria digital"
   - Query B: criar 4–5 palavras-chave compostas sinônimas ou correlatas mantendo o contexto técnico.
       Ex.: "validar presença paciente", "registro falta atendimento", "identificação impressão digital"
   - Evitar palavras genéricas (paciente, sistema, consulta) e palavras de ligação (de, para, em).
5) Executar RAG com Query A e Query B (sempre duas buscas distintas)
6) Selecionar melhores trechos (até 3 frases) — requisito pode aparecer em pequeno trecho do RAG
7) Comparar requisito vs snippets usando pensamento crítico
8) Classificar:
   - TIPO: Funcional (O QUE) / Nao_funcional (COMO)
   - OBRIGATORIEDADE: Obrigatorio / Desejavel
   - NIVEL_SIMILARIDADE: Atende (>=80%), Atende_parcialmente (30–79%), Nao_atende (<30%)
9) Resumir TEXTO_RAG em 2–3 linhas (ou "Não informado")
10) Criar DESCRICAO iniciando com "Deve", "Precisa", "É obrigatório", etc.
11) Salvar com adicionar_requisito usando formato exato:

numero: "X", modulo: "X", funcionalidade: "X", texto_rag: "X", descricao: "X", tipo: "X", obrigatoriedade: "X", nivel_similaridade: "X"

⚠️ NÃO usar JSON, não mudar formato, não deixar campos vazios.
"""
    return system_message_prompt


def query_analise():
    query = """
PROCESSAR REQUISITOS AUTOMATICAMENTE (1 a 100)

Para cada requisito:
1. BUSCAR_REQUISITOS(numero)
2. MÓDULO = texto antes de ':'
3. FUNCIONALIDADE = texto após ':'
4. Criar Query A: 2–3 palavras-chave compostas (bigram/trigram) técnicas e específicas do requisito (sem genéricos).
5. Criar Query B: 4–5 palavras-chave compostas sinônimas ou correlatas ao contexto técnico.
6. Executar RAG com Query A e Query B (sempre duas buscas distintas)
7. Escolher melhores trechos (até 3 frases) de cada RAG
8. Comparar requisito vs snippets
9. Classificar TIPO, OBRIGATORIEDADE, NIVEL_SIMILARIDADE
10. Resumir TEXTO_RAG e criar DESCRICAO
11. Salvar com adicionar_requisito no formato:

numero: "X", modulo: "X", funcionalidade: "X", texto_rag: "X", descricao: "X", tipo: "X", obrigatoriedade: "X", nivel_similaridade: "X"

Observações:
- Aceitar requisito presente em pequeno trecho do RAG
- Usar sempre 5 RAGs diferentes antes da comparação
- Queries não podem conter palavras genéricas ou de ligação
- Progresso: "✅ Requisito X processado"
"""
    return query

# criar banco de dados com trt_base
def banco_doc_base():
    vectorstore = ensure_chroma(
        path_pdf="sata/TRT_BASE.pdf",
        persist_dir="sata/BANCO_BASE",
        tokens_size=150,
        tokens_overlap=0,
        modelo="gpt-4o-mini"
    )
    return vectorstore

# buscar funcinalidade similar
"aqui eu ja fiz faz uma tool"
def rag_banco_base(query: str):
    
    vectorstore = banco_doc_base()
    retriever_base = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    docs = retriever_base.get_relevant_documents(query)
    return [page.page_content for page in docs]

# adicionar a analise na raiz do projeto em formato csv
"aqui eu ja fiz faz uma tool"
def adicionar_requisito(input_string):
    """Função que aceita string e faz parse manual dos parâmetros"""
    try:
        print(f"DEBUG - Input recebido: {input_string}")
        
        # Parse manual da string de parâmetros - REGEX CORRIGIDA
        params = {}
        
        # Nova regex mais robusta para capturar parâmetros nome="valor"
        pattern = r'(\w+)\s*=\s*"([^"]+)"'
        matches = re.findall(pattern, input_string)
        
        for key, value in matches:
            params[key] = value
        
        print(f"DEBUG - Parâmetros extraídos: {params}")
        
        # VALIDAÇÃO CRÍTICA - Verificar se todos os campos obrigatórios estão presentes
        campos_obrigatorios = ['numero', 'modulo', 'funcionalidade', 'funcionalidade_similar', 
                              'descricao', 'tipo', 'obrigatoriedade', 'nivel_similaridade']
        
        campos_faltando = [campo for campo in campos_obrigatorios if campo not in params]
        
        if campos_faltando:
            return f"❌ ERRO: Campos obrigatórios faltando: {campos_faltando}"
        
        # Validação específica para nivel_similaridade
        if params.get('nivel_similaridade') not in ['Atende', 'Atende_parcialmente', 'Nao_atende']:
            return f"❌ ERRO: nivel_similaridade deve ser 'Atende', 'Atende_parcialmente' ou 'Nao_atende'. Recebido: '{params.get('nivel_similaridade')}'"
        
        arquivo = "analise.csv"
        
        # Cria arquivo com cabeçalho se não existir
        if not os.path.exists(arquivo):
            with open(arquivo, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow([
                    "Numero",
                    "Modulo", 
                    "Funcionalidade",
                    "Funcionalidade_Similar_TRT",
                    "Descricao_Requisito",
                    "Tipo_Requisito",
                    "Obrigatoriedade",
                    "Nivel_Similaridade"
                ])
        
        # Extrai dados dos parâmetros
        linha = [
            params.get("numero", ""),
            params.get("modulo", "Sistema"),
            params.get("funcionalidade", ""),
            params.get("funcionalidade_similar", ""),
            params.get("descricao", "")[:300] + "..." if len(params.get("descricao", "")) > 300 else params.get("descricao", ""),
            params.get("tipo", ""),
            params.get("obrigatoriedade", ""),
            params.get("nivel_similaridade", "")  # ESTE CAMPO AGORA SERÁ PREENCHIDO
        ]
        
        # Adiciona ao CSV
        with open(arquivo, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow(linha)
        
        return f"✅ SUCESSO: Requisito {linha[0]} adicionado ao CSV com nivel_similaridade: {params['nivel_similaridade']}!"
    
    except Exception as e:
        return f"❌ ERRO: {str(e)} | Input: {input_string[:100]}..."
    

def verificar_ultimas_linhas(vazio: str):
    try:
        # Detecta automaticamente o delimitador
        with open('analise.csv', 'r', encoding='utf-8') as f:
            amostra = f.read(2048)  # lê um pedaço para análise
            f.seek(0)
            dialect = csv.Sniffer().sniff(amostra, delimiters=[',', ';', '\t'])
            sep_detectado = dialect.delimiter

        # Agora lê com pandas usando o separador detectado
        df = pd.read_csv(
            'analise.csv',
            sep=sep_detectado,
            quotechar='"',
            engine='python'
        )

        if len(df) >= 2:
            return df.tail(2).to_string(index=False)
        elif len(df) == 1:
            return df.tail(1).to_string(index=False)
        else:
            return "Arquivo vazio. Nenhum requisito processado ainda."

    except FileNotFoundError:
        return "Arquivo não encontrado. Iniciando do zero."