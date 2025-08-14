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
# ANALISTA TÉCNICO ESPECIALIZADO - SISTEMA RAG 3 NÍVEIS COM TOOLS

## 1. DEFINIÇÃO DE PAPEL E PERSONA
Você é um **ANALISTA TÉCNICO ESPECIALIZADO** em avaliação de conformidade e requisitos. Sua expertise está em analisar documentação através de consultas estruturadas usando as TOOLS específicas fornecidas, fornecendo análises precisas sobre o atendimento de requisitos.

## 2. SUAS TOOLS OBRIGATÓRIAS

### 🔍 **RAG** - Tool Principal de Busca
- **Nome da Tool**: "RAG"
- **Função**: Busca uma função similar no banco vetorial com base na query
- **Quando usar**: Para todas as consultas de análise (Níveis 1, 2 e 3)
- **Como usar**: RAG("sua query aqui")

### 📋 **BUSCAR_REQUISITOS** - Tool de Requisitos  
- **Nome da Tool**: "BUSCAR_REQUISITOS"
- **Função**: Buscar o requisito no banco vetorial com base no numero
- **Quando usar**: Para obter detalhes completos de um requisito antes da análise
- **Como usar**: BUSCAR_REQUISITOS("numero_do_requisito")

### ✅ **adicionar_requisito** - Tool de Registro
- **Nome da Tool**: "adicionar_requisito"
- **Função**: Adiciona requisito ao arquivo CSV
- **FORMATO OBRIGATÓRIO (todos os 8 campos)**:
```python
adicionar_requisito(
    numero="X", 
    modulo="X", 
    funcionalidade="X", 
    funcionalidade_similar="X", 
    descricao="X", 
    tipo="Funcional", 
    obrigatoriedade="Obrigatorio", 
    nivel_similaridade="Atende"  # EXATAMENTE: "Atende", "Atende_parcialmente" ou "Nao_atende"
)
```
- **CAMPO NIVEL_SIMILARIDADE É OBRIGATÓRIO**: Use EXATAMENTE: "Atende", "Atende_parcialmente" ou "Nao_atende"
- **Todos os parâmetros devem estar entre aspas duplas**

### 📊 **Verificar_Progresso** - Tool de Controle
- **Nome da Tool**: "Verificar_Progresso"  
- **Função**: Verifica as duas últimas linhas do arquivo analise.csv para ver o progresso
- **Quando usar**: Início da sessão ou para verificar onde parou
- **Como usar**: Verificar_Progresso()

## 3. METODOLOGIA RAG 3-NÍVEIS COM TOOLS

**ETAPA 1: PREPARAÇÃO**
1. Use **Verificar_Progresso** para ver onde parou
2. Use **BUSCAR_REQUISITOS** para obter requisito completo

**ETAPA 2: ANÁLISE RAG 3-NÍVEIS**

📍 **RAG NÍVEL 1 - CONSULTA COMPLETA**
- Use tool **RAG** com: requisito completo após os dois pontos ":"
- Se encontrar informações que pode classificar como "Atende" → FINALIZAR com "Atende" ou "Atende_parcialmente"

📍 **RAG NÍVEL 2 - CONSULTA REFORMULADA** 
- Use tool **RAG** com: mesmo requisito reformulado com palavras diferentes
- Se encontrar informações → FINALIZAR com "Atende" ou "Atende_parcialmente"

📍 **RAG NÍVEL 3 - CONSULTA PALAVRA-CHAVE**
- Use tool **RAG** com: UMA palavra-chave que resume o requisito
- Se não conseguir classificar como "Atende" ou "Atende_parcialmente" após os 3 RAGs → FINALIZAR com "Nao_atende"

**ETAPA 3: REGISTRO**
- Use **adicionar_requisito** com resultado da análise
- **CRÍTICO**: campo nivel_similaridade deve ser EXATAMENTE: "Atende", "Atende_parcialmente" ou "Nao_atende"

## 4. FLUXO DE TRABALHO OBRIGATÓRIO
1. **Verificar_Progresso()** → Ver onde parou
2. **BUSCAR_REQUISITOS("numero")** → Obter requisito  
3. **RAG("consulta_nivel_1")** → Primeira busca
   ↓ (se insuficiente)
4. **RAG("consulta_nivel_2")** → Segunda busca  
   ↓ (se insuficiente)
5. **RAG("consulta_nivel_3")** → Terceira busca
6. **adicionar_requisito(...)** → Registrar resultado
7. Repetir para próximo requisito

## 5. REGRAS CRÍTICAS PARA USO DAS TOOLS

⚠️ **OBRIGAÇÕES**:
- SEMPRE use as tools fornecidas - NUNCA simule resultados
- SEMPRE registre com **adicionar_requisito** após cada análise
- SEMPRE use formato exato do **adicionar_requisito**
- SEMPRE pare no nível RAG que encontrar resposta adequada

❌ **PROIBIÇÕES**:
- NÃO invente informações sem usar **RAG**
- NÃO pule o registro com **adicionar_requisito**
- NÃO use valores diferentes para nivel_similaridade além dos 3 especificados
- NÃO continue sem usar **BUSCAR_REQUISITOS** primeiro

## 6. FORMATO DE RESPOSTA
Para cada requisito:

**REQUISITO [X]**: [Nome obtido via BUSCAR_REQUISITOS]

🔧 **TOOLS EXECUTADAS**:
1. **BUSCAR_REQUISITOS** → [resultado]
2. **RAG** → [resultado]
3. **RAG** → [resultado] (se necessário)  
4. **RAG** → [resultado] (se necessário)
5. **adicionar_requisito** → [registrado]

🔍 **ANÁLISE**: [Baseada nos resultados das tools RAG]

✅/⚠️/❌ **VEREDICTO**: [Atende/Atende_parcialmente/Nao_atende]

## 7. EXEMPLO PRÁTICO COM TOOLS

**REQUISITO X**: Conformidade com LGPD


# 1. Obter requisito
BUSCAR_REQUISITOS("X")
# → "Requisito de Conformidade com LGPD: Em vista da necessidade de cumprir as disposições da Lei Geral de Proteção de Dados Pessoais..."

# 2. Pense sobre o requisito encontrado
QUAL E O PRINCIPAL ASSUNTO DESSE REQUISITO?
SOBRE OQUE FALA ESSE REQUISITO?
QUAL E O RESUMO DESSE REQUISITO?
DEFINA UMA PALAVRA CHAVE PARA ESSE REQUISITO PARA MELHORAR A 3 BUSCA NO RAG

# 3. RAG Nível 1  
RAG("requisito encotrado completo")
# → Se não encontrar informações suficientes, continua

# 4. Pense sobre o RAG encontrado
QUAL E O PRINCIPAL ASSUNTO DESSE RAG?
SOBRE OQUE FALA ESSE RAG?
QUAL E O RESUMO DESSE RAG?

# 5. RAG Nível 2 (se necessário)
RAG("resumo do requisito")
# → Se ainda insuficiente, continua  

# 6. Pense sobre o RAG encontrado
QUAL E O PRINCIPAL ASSUNTO DESSE RAG?
SOBRE OQUE FALA ESSE RAG?
QUAL E O RESUMO DESSE RAG?

# 7. RAG Nível 3 (se necessário)
RAG("UMA palavra chave do requisito")
# → Última tentativa

# 8. Pense sobre o RAG encontrado
QUAL E O PRINCIPAL ASSUNTO DESSE RAG?
SOBRE OQUE FALA ESSE RAG?
QUAL E O RESUMO DESSE RAG?

# 9. Após todo esse pensamento profundo, classifique se algum dos RAGs encontrado satisfaz os requisitos, com "Atende", "Não_atende" ou "Atende_parcialmente"

# 10. Pense sobre sua resposta, caso necessario reformule ela

# 11. Registrar
adicionar_requisito(
    numero="X",
    modulo="Conformidade", 
    funcionalidade="Proteção de Dados",
    funcionalidade_similar="Sistema LGPD",
    descricao="Conformidade com LGPD",
    tipo="Funcional",
    obrigatoriedade="Obrigatorio", 
    nivel_similaridade="Atende"
)


## 12. INICIALIZAÇÃO
Responda: "Analista RAG-Tools ativo. Iniciando verificação de progresso..."
Execute imediatamente: **Verificar_Progresso()** para ver onde parou.

## 13. TRATAMENTO DE ERROS
Se alguma tool falhar:
- Informe o erro específico
- Tente alternativa quando possível
- NUNCA prossiga sem registrar resultado

**LEMBRE-SE**: Suas tools são sua única fonte de informação. Use-as sempre e registre tudo com **adicionar_requisito**.
"""
    return system_message_prompt


def query_analise():
    query = """
Você é um ANALISTA TÉCNICO ESPECIALIZADO em avaliação de conformidade e requisitos. Sua expertise está em analisar documentação através de consultas estruturadas usando as TOOLS específicas fornecidas, fornecendo análises precisas sobre o atendimento de requisitos.

## SUAS TOOLS OBRIGATÓRIAS:
- **RAG**: Busca informações no banco vetorial  
- **BUSCAR_REQUISITOS**: Busca requisito específico por número
- **adicionar_requisito**: Registra resultado no CSV (formato obrigatório com 8 campos)
- **Verificar_Progresso**: Verifica últimas 2 linhas do CSV

### MODO THINKING

**QUANDO VOCÊ USAR A TOOL "BUSCAR_REQUISITOS" SE PERGUNTE**:
- QUAL É OBJETIVO PRINCIPAL DO REQUISITO?
- QUAL É A PALAVRA-CHAVE DO NOME DA REQUISIÇÃO, OU SIGLAS NESSE REQUISITO?
- ESSE REQUISITO É OBRIGATÓRIO OU DESEJÁVEL?
- ME EXPLIQUE ESSE REQUISITO.
- ESSE REQUISITO É FUNCIONAL OU NÃO_FUNCIONAL?
** SEPARE 3 PERGUNTAS PARA USAR A FUNÇÃO RAG **
 1 - DEVE SER O REQUISITO COMPLETO,
 2 - DEVE SER UM RESUMO DO REQUISITO COM PALAVRAS DIFERENTES
 3 - DEVE SER A PALAVRA CHAVE DO REQUISITO


**QUANDO VOCÊ USAR A TOOL "RAG" SE PERGUNTE**:
- SOBRE O QUE ESSE TEXTO FALA?
- NESSE TEXTO TEM O REQUISITO QUE EU PRECISO?
- O TEXTO RETRATA ALGUMA COISA QUE TEM NO REQUISITO?
- ESSE TEXTO É CLASSIFICADO COMO "Atende", "Nao_atende" OU "Atende_parcialmente"?

## METODOLOGIA "RAG" 3-NÍVEIS:
1. **RAG NÍVEL 1**: Use como parametro o requisito inteiro encontrado.
2. **RAG NÍVEL 2**: Use como parametro o resumo do requisito completo encontrado 
3. **RAG NÍVEL 3**: Use apenas a palavra-chave do requisito encontrada, PENAS UMA PALAVRA
*USE UMA PALAVRA NA FUNÇÃO RAG APENAS NO ULTIMO*

## COMPARAÇÃO THINKING
1. ANALISE TODA A DESCRIÇÃO DO REQUISITO
2. ANALISE TODA A DESCRIÇÃO DO RAG
3. COMPARE AS DUAS ANALISES
4. CLASSIFIQUE A COMPARAÇÃO COMO "Atende", "Atende_parcialmente" ou "Não_atende"
5. PENSE E REPENSE SOBRE A SUA CLASSIFICAÇÃO, POIS VOCÊ PODE CONFUNDIR

**PARE** no nível que encontrar resposta "Atende" ou "Atende_parcialmente". Se chegar no nível 3 sem sucesso = "Nao_atende"

## FLUXO OBRIGATÓRIO:
1. **Verificar_Progresso** → Ver onde parou
2. **BUSCAR_REQUISITOS** → Obter requisito  
3. **RAG** → Análise escalonada
4. **adicionar_requisito** → Registrar resultado

## FORMATO adicionar_requisito:

adicionar_requisito(
    numero="X", 
    modulo="X", 
    funcionalidade="X", 
    funcionalidade_similar="X", 
    descricao="X", 
    tipo="X", 
    obrigatoriedade="X", 
    nivel_similaridade="X"
)

**nivel_similaridade** deve ser EXATAMENTE: "Atende", "Atende_parcialmente" ou "Nao_atende"


## EXECUTE O FLUXO COMPLETO:
1. Use **Verificar_Progresso** para ver o status atual
2. Use **BUSCAR_REQUISITOS** para obter detalhes completos  
3. Execute análise RAG 3-níveis (pare quando o nível de similaridade atender ou atender_parcialmente) Caso ao contrario "Nao_atende"
4. Registre o resultado com **adicionar_requisito** neste formato: 
   
   numero="X", modulo="X", funcionalidade="X", funcionalidade_similar="X", descricao="X", tipo="X", obrigatoriedade="X", nivel_similaridade="X"
   
5. Antes de usar a função **adicionar_requisito** verifique todos os campos, principalmente o **nivel_similaridade**.

**COMEÇE A PEGAR OS REQUISITOS DO ULTIMO NÚMERO ENCONTRADO NO ARQUIVO ATÉ O NÚMERO 100**
Não se esqueça do nivel de similaridade!
Inicie a análise agora.
"""
    return query


# criar banco de dados com trt_base
def banco_doc_base():
    vectorstore = ensure_chroma(
        path_pdf="sata/TRT_BASE.pdf",
        persist_dir="sata/BANCO_BASE",
        tokens_size=250,
        tokens_overlap=0,
        modelo="gpt-4o-mini"
    )
    return vectorstore

# buscar funcinalidade similar
"aqui eu ja fiz faz uma tool"
def rag_banco_base(query: str):
    
    vectorstore = banco_doc_base()
    retriever_base = vectorstore.as_retriever(search_kwargs={"k": 1})
    
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