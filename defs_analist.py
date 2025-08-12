from vectorstore import ensure_chroma
from typing import Dict
import os, json , csv, re

#====== ANALISE ENTRE DOCUMENTOS ======##

# prompt da analise
def prompt_analista():
    system_message_prompt = """
    Voce e um assistente especialista em Analise de Editais. Sua missao e processar requisitos sequencialmente e salvar no CSV.

    FERRAMENTAS DISPONIVEIS:
    1) BUSCAR_REQUISITOS - busca requisito pelo numero
    2) RAG - busca funcionalidades similares   
    3) adicionar_requisito - salva analise no CSV

    ALGORITMO SIMPLES:
    1. Comece com numero 1
    2. Para cada requisito:
    - Use BUSCAR_REQUISITOS para pegar o requisito
    - Use RAG para encontrar uma funcionalidade similar ao requisito
    - Use adicionar_requisito para salvar
    3. Continue ate nao encontrar mais requisitos

    IMPORTANTE - COMO USAR adicionar_requisito CORRETAMENTE:

    A ferramenta adicionar_requisito tem um problema: ela NAO aceita JSON como string.
    Voce deve passar os dados como dicionario Python diretamente.

    FORMATO CORRETO:
    Action: adicionar_requisito
    Action Input: numero: "1", modulo: "gestão", funcionalidade: "Sistema de gestão ", funcionalidade_similar: "Gestao ISSQN TRT", descricao: "Texto do requisito", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende"

    NAO USE:
    - Aspas simples ou duplas ao redor de todo o input
    - Chaves no inicio e fim do input
    - Formato JSON string

    ###### REGRAS DE PREENCHIMENTO ######
    #### COMO PREENCHER O NUMERO ####
    - Você deve serguir uma sequencia de 1 - ate acabar os requisitos
    - Cada requisito deve ter uma numeração sequencial
    - Sempre seguir a sequencia numeral

    #### COMO PREENCHER O MODULO ####
    - Para você preencher o modulo você terá que indentificar todas as palavras contidas no requisito antes dos dois pontos ':'
    - Sempre pegar as palavras antes dos dois pontos, indetificado no requisito.
    - Vamos dar um Exemplo - se o requisito exatraido foi: 6,"Requisito de Responsabilidade: O licitante responsabilizar-se-á exclusiva e formalmente pelas transações efetuadas em seu nome, assume como firmes e verdadeiras suas propostas e seus lances.". Você terá que pegar a parte "Requisito de Responsabilidade" como o modulo.

    #### COMO PREENCHER A FUNCIONALIDADE ####
    - Para você preencher a funcionalidade, você terá que preencher com o requisito resgato todo, sem os numeros, e sem o nome do modulo.
    - Coloque apenas o requisito, tudo depois do dois pontos ":"
    - Exemplo, se temos o requisito: 14,"Requisito de Documentação: Os licitantes encaminharão, exclusivamente por meio do sistema, proposta com a descrição do objeto ofertado e o preço, até a data e o horário estabelecidos para Recebimento das Propostas.". Você deverá resgatar apenas a parte depois dos dois pontos: " Os licitantes encaminharão, exclusivamente por meio do sistema, proposta com a descrição do objeto ofertado e o preço, até a data e o horário estabelecidos para Recebimento das Propostas"

    #### COMO PREENCHER A SIMILARIDADE SIMILAR ####
    - Para prencher a fucionalidade similar você deve pegar a funcionalidade similar resgatada pela função RAG, e resumir ela
    - Busque na funcionalidade similar os pontos fundamentais para se usar, como as funcionalidades principais.
    - Deixe o tamanho do resumo da funcionalidade similar, semelhante ao requisito.
    - Caso não tenha funcionalidades similares, preencha com "Não informado"


    #### COMO PREENCHER A DESCRIÇÃO ####
    - Para preencher a descrição você tera que fazer um resumo, de que o requisito precisa.
    - Use paravras como: "Deve","Precisa","Tem que","Necessita","É obrigado a","Carece de","Convém","É adequado"
    "Exige","Requer","Demanda". Pegue a mais adequada para a ocasião.

    #### COMO PREENCHER O TIPO #### 
    - Primeiro você terá que fazer uma leitura do requisito
    - Após a leitura, você teŕa que classificar ele como "FUNCIONAL" ou "NÃO FUNCIONAL"
    - Quando devé ser "FUNCIONAL"?: Quando o requisito descreve o que o sistema, produto ou serviço deve fazer — suas funções, comportamentos, operações e interações. Geralmente está ligado a ações específicas, resultados esperados ou processos de negócio que precisam ser implementados.
    - Quando devé ser "NÃO FUNCIONAL"?:Quando o requisito descreve como o sistema, produto ou serviço deve executar suas funções, estabelecendo restrições, padrões de qualidade, desempenho, segurança, usabilidade ou conformidade, sem acrescentar novas funcionalidades.

    #### COMO PREENCHER A OBRIGATORIEDADE ####
    - Primerio indentifique o requisito, analise ele
    - Após a analise do requisito você poderá indentificar ele como "OBRIGATORIO" ou "DESEJAVEL"
    - Deve ser "OBRIGATORIO" quando: O requisito é indispensável para o funcionamento correto do sistema, produto ou serviço, atendendo a necessidades essenciais, normas, legislações ou critérios definidos como mandatórios no projeto ou na licitação. Sua ausência compromete diretamente o atendimento aos objetivos principais.
    - Deve ser "DESEJAVEL" quando: O requisito acrescenta valor, melhoria ou conveniência ao sistema, produto ou serviço, mas não é indispensável para seu funcionamento básico. Pode ser implementado como diferencial, otimização ou aprimoramento, sem comprometer a entrega mínima caso não seja atendido.


    #### COMO PREENCHER O NIVEL DE SIMILARIDADE ####
    - Para preencher o nivel de similaridade você deve analisar o requisito juntamente com a funcionalidade similar, e classificar,
    entre "Atende", "Não atende" e "Atende parcialmente".
    %Regras%
    

    ########### /REGRAS DE PREENCHIMENTO #######



    CAMPOS OBRIGATORIOS (8 campos):
    - numero: numero do requisito (sempre entre aspas)
    - modulo: no requisito extraido, todas as palavras que estão antes dos dois pontos ":"
    - funcionalidade: nome da funcionalidade requisitada
    - funcionalidade_similar: resumo da funcionalidade encontrada no rag
    - descricao: Explicação completa do requisito.
    - tipo: tipo do requisito
    - obrigatoriedade: obrigatoriedade do requisito
    - nivel_similaridade: classificação do nivel similar do requisito com a funcionalidade similar resgatada

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
    Action Input: numero: "2", modulo: "Módulo Nota Fiscal de Serviços Eletrônica Eventual – NFS-e Eventual", funcionalidade: "Emissão de notas fiscais eventuais", funcionalidade_similar: "Permitir que seja emitida a NFS-e avulsa de um serviço eventual (7.1.10)", descricao: "O sistema deverá permitir ao contribuinte a emissão de notas fiscais eventuais. Entende-se como nota fiscal eventual aquela que o prestador poderá emitir para atividades não cadastradas em sua base de dados junto a Prefeitura de Juiz de Fora.", tipo: "Funcional", obrigatoriedade: "Desejavel", nivel_similaridade: "Atende"
    Action Input: numero: "3", modulo: "Domicílio Eletrônico Tributário - DET", funcionalidade: "Canal de Comunicação com Contribuinte", funcionalidade_similar: "–", descricao: "O sistema deverá contar com canal de comunicação para envio de notificações, autos de infrações e avisos ao contribuinte e seus retornos, atendendo aos preceitos legais pertinentes ao DET.", tipo: "Não funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Não Atende"

    ATENÇÃO! ESSES EXEMPLOS SÃO FICTICIOS, USE APENAS O FORMATO DELES(ESQUELETO) COMO REFERENCIA, NÃO USE NENHUM TIPO DE DADOS COMO BASE, SÃO INVENTADOS.

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
    Action Input: requisito encontrado pelo - Action: BUSCAR_REQUISITOS

    Action: adicionar_requisito  
    Action Input: numero: "1", modulo: "Sistema", funcionalidade: "Nome funcionalidade", funcionalidade_similar: "Similar TRT", descricao: "Descricao resumida", tipo: "Funcional", obrigatoriedade: "Obrigatorio", nivel_similaridade: "Atende"

    Requisito 2:
    Action: BUSCAR_REQUISITOS
    Action Input: numero 2

    E assim por diante...

    LEMBRE-SE: O Action Input da ferramenta adicionar_requisito NAO e JSON. E uma lista de parametros separados por virgula.
    PREENCHER O CAMPO DE NIVEL DE SIMILARIDADE E OBRIGATORIO!
    MUITA ATENÇÃO ANTES DE MEDIR SIMILARIADE!!!! VERIFIQUE CORRETAMENTE A SEMELHANÇA ENTRE O REQUISITO E A FUNCIONALIDADE SIMILAR ANTES DE PREENCHER! 
    """
    return system_message_prompt

# prompt da query(analise)
def query_analise():
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
    numero: "X", modulo: "X", funcionalidade: "X", funcionalidade_similar: "X", descricao: "X", tipo: "X", obrigatoriedade: "X", nivel_similaridade: "X"

    IMPORTANTE: NAO use JSON string. Use lista de parametros separados por virgula. 
    LEMBRE-SE DE COLOCAR O NIVEL DE SIMILARIDADE!!!!
    BUSQUE REQUISITOS ATÉ O NUMERO 300
    INICIAR AGORA.

    """
    return query

# criar banco de dados com trt_base
def banco_doc_base():
    vectorstore = ensure_chroma(
        path_pdf="sata/TRT_BASE.pdf",
        persist_dir="sata/BANCO_BASE",
        tokens_size=100,
        tokens_overlap=20,
        modelo="gpt-4o-mini"
    )
    return vectorstore

# buscar funcinalidade similar
"aqui eu ja fiz faz uma tool"
def rag_banco_base(query: str):
    vectorstore = banco_doc_base()
    rag = vectorstore.similarity_search(query, k=1)
    return [page.page_content for page in rag]

# adicionar a analise na raiz do projeto em formato csv
"aqui eu ja fiz faz uma tool"
def adicionar_requisito(input_string):
    """Função que aceita string e faz parse manual dos parâmetros"""
    try:
        print(f"DEBUG - Input recebido: {input_string[:100]}...")
        
        # Parse manual da string de parâmetros
        params = {}
        
        # Regex para capturar parâmetros nome="valor"
        pattern = r'(\w+)="([^"]*)"'
        matches = re.findall(pattern, input_string)
        
        for key, value in matches:
            params[key] = value
        
        print(f"DEBUG - Parâmetros extraídos: {params}")
        
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
            params.get("nivel_similaridade", "")
        ]
        
        # Adiciona ao CSV
        with open(arquivo, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow(linha)
        
        return f"✅ SUCESSO: Requisito {linha[0]} adicionado ao CSV!"
    
    except Exception as e:
        return f"❌ ERRO: {str(e)}"
        
    except Exception as e:
        return f"❌ ERRO: {str(e)} | Input: {input_string[:100]}..."
