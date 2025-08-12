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
🚨 ATENÇÃO CRÍTICA - PRIMEIRA AÇÃO OBRIGATÓRIA
ANTES DE FAZER QUALQUER COISA, VOCÊ DEVE OBRIGATORIAMENTE USAR A TOOL "Verificar_Progresso"!!!
NÃO EXISTE EXCEÇÃO! SEMPRE COMECE COM ESTA TOOL!

🔧 FERRAMENTAS DISPONÍVEIS - DETALHAMENTO COMPLETO E OBRIGATÓRIO:
🔍 TOOL 1: Verificar_Progresso
QUANDO USAR:

PRIMEIRA AÇÃO SEMPRE - antes de qualquer processamento
Para retomar trabalho interrompido
Para verificar continuidade de análises

COMO USAR:
Action: Verificar_Progresso
Action Input: [sem parâmetros]
O QUE RETORNA:

Último número de requisito processado
Status da análise atual
Próximo número a ser processado

EXEMPLO PRÁTICO:
Resultado: "Última análise: requisito 54. Próximo a processar: requisito 55"

🔎 TOOL 2: BUSCAR_REQUISITOS
QUANDO USAR:

Para extrair requisito específico do documento CONCORRENTE
Sempre após verificar progresso
Um por vez, sequencialmente

COMO USAR:
Action: BUSCAR_REQUISITOS
Action Input: numero [X]
ONDE [X] = número sequencial do requisito (33, 34, 35, etc.)
O QUE RETORNA:

Texto completo do requisito
Estrutura: "Número, Módulo: Funcionalidade"
Conteúdo integral extraído do CONCORRENTE

EXEMPLO PRÁTICO:
Input: numero 5
Output: "5,"Requisito de Segurança: O sistema deve implementar autenticação de dois fatores para todos os usuários administrativos, seguindo padrões NIST e com logs de auditoria completos."
REGRAS CRÍTICAS:

Use apenas números sequenciais: 99, 100, 101, 102, 103...
NÃO pule números - sempre sequencial
Se retornar vazio = fim dos requisitos


🧠 TOOL 3: RAG (Retrieval-Augmented Generation)
QUANDO USAR:

IMEDIATAMENTE APÓS cada BUSCAR_REQUISITOS
Para encontrar funcionalidade similar no TRT_BASE
Busca semântica por embeddings vetoriais

COMO USAR:
Action: RAG
Action Input: [texto completo do requisito encontrado pelo BUSCAR_REQUISITOS + palavrea chave do requisito]
O QUE RETORNA:

Funcionalidade mais similar do TRT_BASE
Texto detalhado da funcionalidade encontrada
Contexto técnico da solução existente

EXEMPLO PRÁTICO:
Input: "O sistema deve implementar autenticação de dois fatores, autenticação de dois fatores"
Output: "MÓDULO DE SEGURANÇA TRT - Item 6.4.2: Sistema possui autenticação multifator com token SMS, biometria digital e validação por aplicativo móvel. Inclui logs de auditoria em tempo real e integração com Active Directory."
REGRAS CRÍTICAS:

Sempre que pegar o requisito você deverá buscar uma funcionalidade similar com a função RAG, você vai colocar como parametro o requisito + a paralavra chave do requisito
Passe o texto COMPLETO do requisito + a palavra chave dele destacada, exemplos:
"Requisito de Segurança: As licitantes devem comprovar sua implementação por meio de certificado de auditoria acreditada, atestando o funcionamento e manutenção do SGSI.", você resgatará a paralavra chave: "manutenção do SGSI".
9,"Requisito Tecnológico: A contratada é responsável pela manutenção, adequação de acessos, carga e balanceamento de dados, elasticidade de recursos de hardware, sistema de backups inteligente e previsão de reestabelecimento do serviço, de acordo com os níveis de SLA de mercado, caso a solução não seja ON PREMISE.",  você resgatará a paralavra chave: "balanceamento de dados, elasticidade de recursos de hardware, sistema de backups inteligente e previsão de reestabelecimento do serviço, de acordo com os níveis de SLA de mercado"

NÃO resuma o input - use texto integral + pavrea chave encontrada no requisito
Se não encontrar similar = funcionalidade será "Não informado"


💾 TOOL 4: adicionar_requisito
QUANDO USAR:

APÓS completar busca com RAG
Para salvar análise completa no CSV
UM requisito por vez

FORMATO CRÍTICO - NÃO É JSON!!!
Action: adicionar_requisito
Action Input: numero: "X", modulo: "Y", funcionalidade: "Z", funcionalidade_similar: "W", descricao: "V", tipo: "T", obrigatoriedade: "O", nivel_similaridade: "S"
PARÂMETROS OBRIGATÓRIOS (8 CAMPOS):
1. numero: (string)

Formato: "77", "78", "79", etc.
SEMPRE entre aspas
Sequencial sem pulos

2. Para cada requisito:
   - Use BUSCAR_REQUISITOS para pegar o requisito
   - Use RAG para encontrar um texto contendo uma funcionalidade similar ao requisito
   - Analise CUIDADOSAMENTE para classificar tipo e obrigatoriedade
   - Compare requisito com o texto da funcionalidade similar contida no texto, para determinar nível de similaridade
   - Use adicionar_requisito para salvar COM TODOS OS CAMPOS PREENCHIDOS
3. Continue até não encontrar mais requisitos

⚠️ FORMATO CRÍTICO PARA adicionar_requisito:
A ferramenta adicionar_requisito tem um problema: ela NÃO aceita JSON como string.
Você deve passar os dados como dicionário Python diretamente.

✅ FORMATO CORRETO OBRIGATÓRIO:
Action: adicionar_requisito
Action Input: numero: "5", modulo: "gestão", funcionalidade: "Sistema de gestão", funcionalidade_similar: "Gestao ISSQN TRT", descricao: "Texto do requisito", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende"

❌ NUNCA USE:
- Aspas simples ou duplas ao redor de todo o input
- Chaves { } no início e fim do input
- Formato JSON string

📋 REGRAS DETALHADAS DE PREENCHIMENTO DOS 8 CAMPOS OBRIGATÓRIOS:

1️⃣ NÚMERO:
- Você deve seguir uma sequência de numeração que o usuario pediu até acabar os requisitos
- Cada requisito deve ter uma numeração sequencial
- SEMPRE entre aspas: "18", "19", "20", etc.
- Sempre seguir a sequência numeral

2️⃣ MÓDULO:
- Para preencher o módulo você terá que identificar TODAS as palavras contidas no requisito ANTES dos dois pontos ':'
- SEMPRE pegar as palavras antes dos dois pontos, identificado no requisito
- Exemplo: Se o requisito extraído foi: 6,"Requisito de Responsabilidade: O licitante responsabilizar-se-á exclusiva e formalmente pelas transações..."
- Você terá que pegar a parte "Requisito de Responsabilidade" como o módulo

3️⃣ FUNCIONALIDADE:
- Para preencher a funcionalidade, você terá que preencher com o requisito resgatado todo, sem os números, e sem o nome do módulo
- Coloque apenas o requisito, tudo DEPOIS do dois pontos ":"
- Exemplo: Se temos o requisito: 14,"Requisito de Documentação: Os licitantes encaminharão, exclusivamente por meio do sistema..."
- Você deverá resgatar apenas a parte depois dos dois pontos: "Os licitantes encaminharão, exclusivamente por meio do sistema, proposta com a descrição do objeto ofertado e o preço, até a data e o horário estabelecidos para Recebimento das Propostas"

4️⃣ FUNCIONALIDADE_SIMILAR:
- Para preencher a funcionalidade similar você deve encontrar no texto funcionalidade similar contida no texto, resgatada pela função RAG, e resumir ela
- Busque no texto da funcionalidade similar contida no texto os pontos fundamentais para se usar, como as funcionalidades principais
- Deixe o tamanho do resumo da funcionalidade similar contida no texto, semelhante ao requisito
- Caso não tenha funcionalidades similares contida no texto, preencha com "Não informado"
- Se RAG retornar funcionalidade contida no texto, faça um RESUMO CONCISO (máximo 2 linhas)
- Manter aspectos principais da funcionalidade contida no texto encontrada

5️⃣ DESCRIÇÃO:
- Para preencher a descrição você terá que fazer um resumo, de que o requisito precisa
- Use palavras como: "Deve", "Precisa", "Tem que", "Necessita", "É obrigatório a", "Carece de", "Convém", "É adequado", "Exige", "Requer", "Demanda"
- Pegue a mais adequada para a ocasião
- Reescrever o requisito em formato de necessidade/obrigação
- Ser objetivo e claro sobre o que o sistema/processo deve fazer

6️⃣ TIPO - ANÁLISE CRÍTICA OBRIGATÓRIA:
✅ FUNCIONAL (O QUE o sistema DEVE FAZER):
- Quando o requisito descreve o que o sistema, produto ou serviço deve fazer — suas funções, comportamentos, operações e interações
- Geralmente está ligado a ações específicas, resultados esperados ou processos de negócio que precisam ser implementados
- Identifique como FUNCIONAL quando o requisito:
  ** Descreve uma AÇÃO específica do sistema
  ** Define um COMPORTAMENTO esperado  
  ** Especifica um PROCESSO de negócio
  ** Estabelece uma FUNCIONALIDADE contida no texto nova
- Exemplos de palavras-chave: processar, calcular, emitir, validar, enviar, receber, armazenar, gerar

❌ NÃO FUNCIONAL (COMO o sistema deve FUNCIONAR):
- Quando o requisito descreve como o sistema, produto ou serviço deve executar suas funções, estabelecendo restrições, padrões de qualidade, desempenho, segurança, usabilidade ou conformidade, sem acrescentar novas funcionalidades
- Identifique como NÃO FUNCIONAL quando o requisito:
  ** Define RESTRIÇÕES ou LIMITAÇÕES
  ** Estabelece critérios de QUALIDADE
  ** Especifica PERFORMANCE ou SEGURANÇA
  ** Define PADRÕES de conformidade
  ** Estabelece PENALIDADES ou SANÇÕES administrativas
- Exemplos de palavras-chave: percentual, prazo, multa, sanção, padrão, norma, conformidade, limite

7️⃣ OBRIGATORIEDADE - ANÁLISE CRÍTICA OBRIGATÓRIA:
🔴 OBRIGATÓRIO:
- Primeiro identifique o requisito, analise ele
- O requisito é indispensável para o funcionamento correto do sistema, produto ou serviço, atendendo a necessidades essenciais, normas, legislações ou critérios definidos como mandatórios no projeto ou na licitação
- Sua ausência compromete diretamente o atendimento aos objetivos principais
- Classifique como OBRIGATÓRIO quando:
  ** É exigido por LEI ou NORMA
  ** É indispensável para funcionamento básico
  ** Tem consequências LEGAIS se não atendido
  ** Usa termos imperativos: "deve", "será", "é obrigatório"
  ** Define SANÇÕES por descumprimento
- 95% dos requisitos de edital são obrigatórios

🟡 DESEJÁVEL:
- O requisito acrescenta valor, melhoria ou conveniência ao sistema, produto ou serviço, mas não é indispensável para seu funcionamento básico
- Pode ser implementado como diferencial, otimização ou aprimoramento, sem comprometer a entrega mínima caso não seja atendido
- Classifique como DESEJÁVEL quando:
  ** Usa termos como: "pode", "é recomendável", "preferencialmente"
  ** Não compromete funcionamento básico
  ** É melhoria ou otimização
  ** Não há penalidade por não atender

#### 8) NÍVEL_SIMILARIDADE - COMPARAÇÃO OBRIGATÓRIA: #### 
Instruções para classificação do campo NÍVEL_SIMILARIDADE

Você recebe dois textos:

- Requisito (geralmente curto, objetivo)
- Funcionalidade similar (texto maior, descritivo, contendo uma possível implementação ou descrição da funcionalidade)

Seu trabalho é comparar os dois e classificar o nível de similaridade entre eles, seguindo estas regras estritas:

---

Critérios para classificação

1. Atende
   - A funcionalidade similar contém o requisito completo, ou seja, o requisito está claramente descrito dentro do texto da funcionalidade.
   - A finalidade, escopo e intenção do requisito estão totalmente contemplados no texto da funcionalidade.
   - A implementação do requisito pode ser feita diretamente a partir da funcionalidade descrita.

2. Atende_parcialmente
   - O requisito está parcialmente descrito na funcionalidade similar, ou há partes importantes do requisito que não estão contempladas.
   - A funcionalidade cobre a mesma área geral, mas o escopo difere e será necessário adaptação ou complemento para cumprir o requisito.

3. Nao_atende
   - O requisito não está contido na funcionalidade similar, ou está ausente de forma clara.
   - A finalidade e o escopo da funcionalidade são diferentes do requisito.
   - Será necessária uma nova implementação para atender o requisito.

---

Regras obrigatórias

- Use somente as três categorias: Atende, Atende_parcialmente, ou Nao_atende.
- Nunca deixe o campo vazio.
- Leia ambos os textos com atenção antes de decidir.
- Se estiver inseguro, prefira revisar o conteúdo ao invés de chutar a classificação.

---

Exemplo rápido

- Requisito: "Sistema deve gerar relatórios mensais em PDF."
- Funcionalidade similar: Texto maior que descreve exatamente como gera relatórios mensais em PDF com detalhes.
- Classificação: Atende

- Requisito: "Sistema deve gerar relatórios mensais em PDF com gráficos interativos."
- Funcionalidade similar: Texto maior que só menciona geração de relatórios mensais em PDF, sem gráficos.
- Classificação: Atende_parcialmente

- Requisito: "Sistema deve gerar relatórios mensais em PDF."
- Funcionalidade similar: Texto maior que fala apenas sobre gráficos em tempo real, sem geração de relatórios.
- Classificação: Nao_atende

 EXEMPLO PRÁTICO COMPLETO DE EXECUÇÃO:

Primeiro 1: Verificar_Progresso
QUANDO USAR:

PRIMEIRA AÇÃO SEMPRE - antes de qualquer processamento
Para retomar trabalho interrompido
Para verificar continuidade de análises

depois...

Requisito X:
Action: BUSCAR_REQUISITOS
Action Input: numero X

Action: RAG
Action Input: texto do requisito encontrado pelo BUSCAR_REQUISITOS + palavra chave do requisito

Action: adicionar_requisito
Action Input: numero: "X", modulo: "Requisitos de Segurança", funcionalidade: "Prevenção Contra Fraude", funcionalidade_similar: "QUANTO AO MODELO DE SEGURANÇA (Item 6.6.4.3)", descricao: "Cada usuário deve ser único no sistema a partir do momento de acesso", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende"

Requisito Y:
Action: BUSCAR_REQUISITOS
Action Input: numero Y

E assim por diante...

🚨 VERIFICAÇÃO FINAL OBRIGATÓRIA:
Antes de usar adicionar_requisito, SEMPRE verifique:
✓ Usou a Tool RAG com o requisito + palavra chave
✓ Todos os 8 campos estão preenchidos
✓ nivel_similaridade NÃO está vazio
✓ Tipo está correto (Funcional/Nao_funcional)
✓ Obrigatoriedade está correta (Obrigatorio/Desejavel)
✓ Formato não é JSON - é lista de parâmetros separados por vírgula

🚀 REGRAS IMPORTANTES:
1. NUNCA coloque aspas simples ou duplas ao redor de todo o Action Input
2. NUNCA use formato JSON string
3. Liste os campos separados por vírgula, cada um com seu valor
4. Mantenha textos curtos para evitar problemas
5. Se um campo for muito longo, resuma mantendo o essencial
6. PREENCHER O CAMPO DE NIVEL DE SIMILARIDADE É OBRIGATÓRIO!

⚡ PROCESSAMENTO:
Quando receber "PROCESSAR TODOS", execute o algoritmo automaticamente.
Processe até não encontrar mais requisitos.
Relate o progresso a cada requisito processado.


### Não use os dados dos exemplos como base hora alguma, são apenas inventados.

LEMBRE-SE: O Action Input da ferramenta adicionar_requisito NÃO é JSON. É uma lista de parâmetros separados por vírgula.
ATENÇÃO!!! PENSE E REPENSE SOBRE SUA RESPOSTA ANTES DE PREENCHER
🚨 REGRA DE OURO - FLUXO CONTÍNUO
NUNCA RETORNE "RESPOSTA FINAL" ATÉ QUE TODOS OS REQUISITOS (1-103) TENHAM SIDO PROCESSADOS!
QUALQUER MENSAGEM DE PROGRESSO É APENAS UM STATUS INTERMEDIÁRIO, NÃO UMA CONCLUSÃO!

🔄 ALGORITMO CORRIGIDO - LOOP INFALÍVEL:
1️⃣ SEMPRE comece com: 
   Action: Verificar_Progresso
   Action Input: ""

2️⃣ SE "Próximo a processar: [N]" ENTÃO:
   Para N de N até 103 FAÇA:
      Action: BUSCAR_REQUISITOS
      Action Input: numero [N]
      
      Action: RAG
      Action Input: [texto completo] + [palavra-chave]
      
      Action: adicionar_requisito
      Action Input: numero: "[N]", ... [todos campos]
      
      Action: Verificar_Progresso  ⚠️ SEMPRE APÓS CADA REQUISITO!
      Action Input: ""

3️⃣ SE BUSCAR_REQUISITOS RETORNAR VAZIO PARA [N]:
   Action: Verificar_Progresso
   Action Input: ""
   SE próximo > 103: ENCERRE COM "PROCESSAMENTO COMPLETO"
s
4️⃣ NUNCA gere "Resposta final" até:
   - Todos requisitos de 1 a 103 processados OU
   - Confirmação de fim por Verificar_Progresso
   - não se esqueca de adicionar a linha csv com a tool
"""
    return system_message_prompt

# prompt da query(analise)
def query_analise():
    query = """
🎯 PROCESSAR TODOS OS REQUISITOS - EXECUÇÃO COMPLETA

Execute análise automática COMPLETA de todos os requisitos do edital seguindo o ALGORITMO SEQUENCIAL.

## INSTRUÇÕES ESPECÍFICAS:
1. *Iniciar pelo número 4 porfavor numero 4, começe pelo numero 4 a coletar os requisitos e vai ate o numero 105, sem parar, porfavor não para até o numero 100 cara porfavor, ate o numero 100 *
2. *Para CADA requisito encontrado:*
   - Buscar texto com BUSCAR_REQUISITOS
   - Encontrar uma função similar com RAG, colocando como input o requisito + palavra chave do requisito
   - *ANALISAR CUIDADOSAMENTE* para classificar tipo e obrigatoriedade
   - *COMPARAR requisito com similar* para determinar nível de similaridade
   - Salvar com adicionar_requisito usando formato correto (NÃO JSON)
3. *Continuar até número 103 ou até não encontrar mais requisitos*
4. *Reportar progresso APÓS CADA requisito processado* (Ex: "✅ Requisito 18 processado")

## ⚠️ FORMATO CRÍTICO OBRIGATÓRIO para adicionar_requisito:
numero: "X", modulo: "X", funcionalidade: "X", funcionalidade_similar: "X", descricao: "X", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende"

## 🔥 ATENÇÃO MÁXIMA PARA:
- *NÍVEL_SIMILARIDADE*: Campo OBRIGATÓRIO - nunca deixe vazio! Use: "Atende", "Atende_parcialmente" ou "Nao_atende"
- *TIPO*: Análise cuidadosa - Funcional vs Nao_funcional
- *OBRIGATORIEDADE*: Análise cuidadosa - Obrigatorio vs Desejavel (95% são obrigatórios)
- *FORMATO*: NÃO usar JSON - usar lista de parâmetros separados por vírgula

## 📋 VERIFICAÇÃO OBRIGATÓRIA antes de cada adicionar_requisito:
✓ Todos os 8 campos preenchidos
✓ nivel_similaridade NÃO está vazio
✓ Tipo correto (Funcional/Nao_funcional)
✓ Obrigatoriedade correta (Obrigatorio/Desejavel)
✓ Formato correto (não JSON)

## 🚀 INICIAR PROCESSAMENTO AGORA
Buscar os requisitos de quando o usuario pediu até 105 sequencialmente.
Preencher TODOS os campos obrigatoriamente.
Nunca deixar nivel_similaridade vazio!
Reportar progresso após cada requisito processado.
geralmente a maioria seram que atende, mas queremos investigar profundamente para ver quais requisitos não nos atende ou atende parcialmente.
ATENÇÃO!!! PENSE E REPENSE SOBRE O NIVEL DE SIMILARIDADE ANTES DE PREENCHER E USE COMO INPUT PARA A TOOL RAG O REQUISITO + PALAVRA CHAVE DELE.
"""
    return query
# criar banco de dados com trt_base
def banco_doc_base():
    vectorstore = ensure_chroma(
        path_pdf="sata/TRT_BASE.pdf",
        persist_dir="sata/BANCO_BASE",
        tokens_size=150,
        tokens_overlap=20,
        modelo="gpt-4o-mini"
    )
    return vectorstore

# buscar funcinalidade similar
"aqui eu ja fiz faz uma tool"
def rag_banco_base(query: str):
    
    vectorstore = banco_doc_base()
    retriever_base = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # Cria o LLM aqui dentro (modelo para reformular a query)
    llm_for_queries = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=retriever_base,
        llm=llm_for_queries
    )
    
    docs = retriever.get_relevant_documents(query)
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