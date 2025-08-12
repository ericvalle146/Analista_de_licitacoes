from langchain_openai import ChatOpenAI
import os
from langchain.prompts import ChatPromptTemplate
from vectorstore import ensure_chroma, ensure_chroma_csv
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#====== EXTRAÇÃO DE REQUISITOS LICITAÇÃO ======##

# carregar modelo
def load_model(model):
    return ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY
)

# Criação banco vetorial da licitação
def banco_doc_licitacao():
    vectorstore = ensure_chroma(
        path_pdf="sata/CONCORRENTE.pdf",
        persist_dir="sata/BANCO_CONCORRENTE",
        tokens_size=15_000,
        tokens_overlap=1000,
        modelo="gpt-4o-mini"
    )
    return vectorstore

# recuperção do todo documento da licitação
def recovery_chunks(vectorstore):
    data = vectorstore.get(include=["documents"])
    return data["documents"]

# prompt para recuperar requisitos
def prompt_rag_structured():
    system_prompt = """
    Você é um agente especializado em extração automática de requisitos/modulos técnicos e de negócio a partir de documentos complexos, como editais, licitações, documentos de especificação e termos de referência. Seu papel é identificar e extrair todos os requisitos/modulos explícitos e claros presentes no texto, que geralmente estão estruturados em módulos e subitens numerados.

    INSTRUÇÕES PRINCIPAIS (leia com atenção e siga estritamente):

    1) Entrada:
    - Você vai receber um BLOCO DE TEXTO para analisar.
    - Não mencione, explique ou faça referência ao mecanismo que forneceu o texto. Trabalhe apenas com o que foi recebido.
    
    2) Objetivo imediato:
    - Extraia até 20 requisitos/modulos distintos do texto recebido, obedecendo estas regras:
        * Se o texto contiver 20 ou mais requisitos/modulos claramente expressos, retorne *20 — priorizando os requisitos/modulos mais explícitos e operacionais.
        * Se o texto contiver menos de 20 requisitos/modulos explícitos, retorne apenas os requisitos/modulos que existem — *não crie ou invente requisitos/modulos adicionais.

    3) Regras de conteúdo e fidelidade:
    - Preserve 100% da redação do requisito tal qual aparece no texto. Não reescreva, não resuma, não interprete.
    - Inclua somente requisitos/modulos EXPLÍCITOS. Não transforme exemplos, observações explicativas ou contextos em requisitos/modulos se eles não estiverem formulados como obrigação/condição/entrega.
    - Preserve referências cruzadas e normativas exatamente como no original.
    - Se um requisito ocupar várias frases ou parágrafos, inclua todas as frases pertinentes em UMA SÓ LINHA (no output), mantendo pontuação e termos normativos.
    - Se o texto apresentar numeração de módulos (como 1.1, 1.2, 2.8, etc.), preserve essa numeração dentro do conteúdo do requisito.

    4) Classificação:
    - SEMPRE, inclua alguma categoria para cada requisito. Não deixe de incluir uma categoria para cada requisito, mesmo que ele não esteja explicitamente mencionado no texto.

    5) Formato de saída (OBRIGATÓRIO — pronto para CSV):
    - Cada requisito deve ser uma linha separada, entre aspas duplas, exatamente neste formato:
        "Categoria: <texto integral do requisito com numeração do módulo se existir>"
    - Não coloque numeradores extras, bullets, cabeçalhos, ou explicações.
    - NADA além dessas linhas deve ser impresso. Qualquer texto adicional é estritamente proibido.

    6) Critérios de seleção quando houver mais de 20 candidatos:
    - Priorize requisitos/modulos que expressem ações, condições, entregas, critérios de aceitação, limites, interfaces, conformidade normativa e obrigações contratuais.
    - Exclua trechos que são meramente descritivos, exemplos ilustrativos, ou informações logísticas não-requisitivas.

    7) Comportamento obrigatório:
    - Preserve a numeração dos módulos (se existir) dentro do texto do requisito
    - Não altere o conteúdo do requisito para melhorar legibilidade — preserve pontuação e termos técnicos.
    - Não inclua metadados, comentários, instruções ao leitor ou marcações internsas.

    8) Exemplo de saída (formato exato — mantenha estritamente o padrão):
    "Requisito de Software: O sistema deverá ser desenvolvido em linguagem de programação Java, com suporte a banco de dados Oracle, e deverá ser desenvolvido em 3 (três) meses, a partir do dia 20 de dezembro de 2023, com início de produção no dia 15 de janeiro de 2024."
    "Requisito de Segurança: O sistema deverá ser desenvolvido com segurança, com mecanismos de autenticação e autorização robustos, e com mecanismos de segurança de dados, como criptografia e proteção de dados em transitória, para garantir a confidencialidade, integridade e disponibilidade dos dados."
    "requisito de Performance: O sistema deverá ser desenvolvido com desempenho otimizado, com recursos de cache, memória e processamento, e com mecanismos de otimização de código e de rede."
    "Modulo de Negócio: A licitante assinalará 'SIM' ou 'NÃO' em campo próprio do sistema eletrônico, relativo às Declarações."
    

    ### REGRA DE AGRUPAMENTO E RESUMO DE MÓDULOS CONSECUTIVOS (2–3) ####

    Objetivo: quando apropriado, agrupar módulos numerados sequencialmente e próximos (ex.: x.y, x.z, x.a, x.b) em UMA ÚNICA LINHA resumida, reduzindo redundância mas preservando todas as obrigações, condições, valores e referências normativas. O agrupamento é permitido apenas para sequência de 2 a 3 módulos consecutivos.

    9) Identificação de módulos "próximos":
    - Não agrupe módulos que mudem o nível superior (ex.: a.a.x com b.a.x NÃO são consecutivos para fins de agrupamento).

    10) Tamanho do grupo:
    - Agrupe entre 2 e 3 módulos por linha.
    - Regra prática para dividir longas sequências: se o módulo x.y der 100 caracteres, o módulo x.y tem 100 caracteres, a junção dos dois terá que dar 150 caracteres para melhor leitura.
    - Regras: Você nunca quando usar o agrupamento deverá preservar todos os requisitos/modulos agrupados juntos na mesma linha, sempre resuma para dar uma linha resumida com todas as informações necessárias.

    11) Quando NÃO agrupar:
    - Se qualquer subitem do conjunto apresentar conflito semântico (condições mutuamente excludentes), prazos diferentes, responsáveis distintos, valores distintos ou restrições incompatíveis — NÃO agrupar; liste separadamente.
    - Se a agrupação implicar perda ou omissão de informações críticas (prazos, valores, limites), NÃO agrupar.
    - Se os subitens incluírem aprovações, penalidades ou exceções com efeitos distintos, NÃO agrupar.

    RESUMO: agrupe módulos consecutivos de 2 módulos a 3 quando pertencerem ao mesmo contexto e não houver conflito; resuma concatenando e preservando o contexto e termos críticos (prazos, valores, referências).

    O que é um módulo? módulo é uma parte do texto que contém o requisito e que se encontra um número representando esse módulo. Exemplo: 3.6 A contratada deve manter a integridade e a integridade do sistema, conforme estabelecido nos tópicos de manutenção.

    #### ATENÇÃO!!! NÃO PRECISA USAR SEMPRE OS EXEMPLOS QUE EU TE ENVIEI!!! É APENAS PARA VOCÊ ENTENDER O MODELO.
    TRABALHE COM MÁXIMA RIGOR, PRECISÃO E FIDELIDADE.
    Ao receber o texto produza APENAS a saída conforme as regras acima.

    ### FORMATO DE SAÍDA: APENAS CONTEÚDO ENTRE ASPAS ###

    NÃO CONFUNDA OS REQUISITOS, ANALISE BEM ANTES DE CATEGORIZAR.
    """

    prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("human", "{input}")
    ])
    return prompt

# query para extrair requisitos
def request_exatraction():
    pergunta = """Por favor, analise o texto abaixo cuidadosamente e extraia no mínimo 20 requisitos/modulos distintos, numerados conforme a estrutura. Para cada requisito, gere um identificador no formato = "tipo do requisito:..." . Liste todos os requisitos/modulos extraídos, separados por aspas (""). O texto deve ser preservando a fidelidade ao texto original e mantendo o contexto integral, sem alterar o sentido ou omitir informações importantes
    #### IMPORTANTE ####
    A SAÍDA DEVE CONTER EXATAMENTE O NÚMERO SEQUENCIAL QUE ESTÁ CONTIDO NA PARTE DO TEXTO EM QUE PEGOU O REQUISITO!!!!
    """
    return pergunta

# dar contexto sequencial para meu prompt 
"aqui você faz uma tool"
def ler_ultimas_linhas_csv(caminho_arquivo="requisitos.csv", num_linhas=4) -> str:
    """
    Lê as últimas num_linhas linhas do CSV e retorna como string concatenada, separadas por nova linha.
    Se o arquivo não existir ou estiver vazio, retorna string vazia.
    """
    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            linhas = f.readlines()
            if not linhas:
                return ""
            ultimas = linhas[-num_linhas:]
            return "".join([linha.strip() + "\n" for linha in ultimas]).strip()
    except FileNotFoundError:
        return ""

# numerar arquivo csv formato aceitavel
"aqui você faz uma tool"
def numerar_arquivo_csv():
    """
    Modifica o arquivo requisitos.csv adicionando numeração sequencial
    Formato de saída: 1,"Requisito de Negócio: texto..."
    """
    
    arquivo_original = "requisitos.csv"
    arquivo_numerado = "requisitos_numerado.csv"
    
    try:
        # Lê o arquivo original
        with open(arquivo_original, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
        
        # Remove linhas vazias
        linhas = [linha.strip() for linha in linhas if linha.strip()]
        
        # Cria arquivo numerado
        with open(arquivo_numerado, 'w', newline='', encoding='utf-8') as f:
            for i, linha in enumerate(linhas, 1):
                # Remove aspas duplas se existirem no início e fim
                linha_limpa = linha.strip().strip('"')
                
                # Escreve no formato: NUMERO,"texto"
                f.write(f'{i},"{linha_limpa}"\n')
        
        print(f"✅ Arquivo numerado criado: {arquivo_numerado}")
        return arquivo_numerado
        
    except Exception as e:
        print(f"❌ Erro ao numerar arquivo: {e}")
        return None

# recuperação de requisitos
"aqui você faz uma tool"
def rag_extration_requisitos(chain, pergunta, partes):
    if not os.path.exists("requisitos.csv"):
        contexto_continuacao = ler_ultimas_linhas_csv("requisitos.csv", num_linhas=4)
        
        for n, parte in enumerate(partes):
            input_do_human = (
                pergunta 
                + "\n\nUse como base de continuação as últimas linhas extraídas anteriormente:\n"
                + contexto_continuacao 
                + "\nJuntamente com os novos requisitos que devem ser extraídos:\nTEXTO: " 
                + parte  
            )
            resposta = chain.invoke({"input": input_do_human})
            append_reqs_to_csv(resposta.content, path="requisitos.csv")
            
            # Atualiza o contexto_continuacao com as últimas 4 linhas extraídas após cada iteração
            ultima_linhas = ler_ultimas_linhas_csv("requisitos.csv", num_linhas=4)
            if ultima_linhas:
                contexto_continuacao = ultima_linhas
            else:
                contexto_continuacao = ""
        
        # 🎯 APÓS FINALIZAR O LOOP, NUMERA OS REQUISITOS
        print("🔄 Finalizando extração e adicionando numeração sequencial...")
        arquivo_numerado = numerar_arquivo_csv()
        
        if arquivo_numerado:
            return f"✅ Extração completa! Arquivo final: {arquivo_numerado}"
        else:
            return "⚠️ Requisitos extraídos, mas erro na numeração."
    
    else:
        return "Requisitos já foram extraídos."

# salvamento de requisitos em csv
"aqui você faz uma tool"
def append_reqs_to_csv(content, path='requisitos.csv'):
    import re, csv
    reqs = re.findall(r'"([^"]+)"', content)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for r in reqs:
            w.writerow([r])

# criação do banco vetorial de requisitos extraidos
"aqui você faz uma tool"
def create_vectorstore_req():
    vectorstore_requisitos = ensure_chroma_csv(
    path_csv="requisitos_numerado.csv",
    persist_dir="sata/Banco_db_req"
)
    return vectorstore_requisitos

# Função para alimentar o agente com os requisitos
"aqui você faz uma tool" 
def alimentacao_req(query: str):
    vectorstore_req = create_vectorstore_req()
    req = vectorstore_req._map_by_id.get(query)
    return req