from vectorstore import ensure_chroma
from typing import Dict
import os, json , csv, re

def banco_doc_base():
    vectorstore = ensure_chroma(
        path_pdf="sata/TRT_BASE.pdf",
        persist_dir="sata/BANCO_BASE",
        tokens_size=300,
        tokens_overlap=30,
        modelo="gpt-4o-mini"
    )
    return vectorstore

def rag_banco_base(query: str):
    vectorstore = banco_doc_base()
    rag = vectorstore.similarity_search(query, k=2)
    return [page.page_content for page in rag]

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
                writer = csv.writer(f)
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
            params.get("descricao", "")[:300] + "..." if len(params.get("descricao", "")) > 300 else params.get("descricao", ""),  # Limita descrição
            params.get("tipo", "Funcional"),
            params.get("obrigatoriedade", "Obrigatorio"),
            params.get("nivel_similaridade", "Nao_Atende")
        ]
        
        # Adiciona ao CSV
        with open(arquivo, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(linha)
        
        return f"✅ SUCESSO: Requisito {linha[0]} adicionado ao CSV!"
        
    except Exception as e:
        return f"❌ ERRO: {str(e)} | Input: {input_string[:100]}..."