import chromadb


# ---------------------------------------------------------------------------
# 1. BASE DE DOCUMENTOS
# ---------------------------------------------------------------------------
DOCUMENTOS = [
    (
        "ti-01",
        "Para redefinir sua senha, acesse a opção 'Esqueci minha senha' na tela de login.",
        {"categoria": "acesso", "tipo": "senha"},
    ),
    (
        "ti-02",
        "Para alterar sua senha, entre nas configurações da conta e escolha a opção de segurança.",
        {"categoria": "acesso", "tipo": "senha"},
    ),
    (
        "ti-03",
        "A rede sem fio está disponível nos laboratórios e pode ser acessada com as credenciais institucionais.",
        {"categoria": "rede", "tipo": "wifi"},
    ),
    (
        "ti-04",
        "Quando a conexão Wi-Fi estiver instável, desligue e ligue novamente o adaptador de rede.",
        {"categoria": "rede", "tipo": "wifi"},
    ),
    (
        "ti-05",
        "Se o computador estiver muito lento, verifique os programas abertos e o espaço disponível no disco.",
        {"categoria": "hardware", "tipo": "desempenho"},
    ),
    (
        "ti-06",
        "Em caso de falha no teclado ou mouse, teste o equipamento em outra porta USB.",
        {"categoria": "hardware", "tipo": "periferico"},
    ),
    (
        "ti-07",
        "Para instalar um programa autorizado, solicite a instalação ao setor de suporte de TI.",
        {"categoria": "software", "tipo": "instalacao"},
    ),
    (
        "ti-08",
        "Antes de instalar atualizações, salve seus arquivos e feche os aplicativos em execução.",
        {"categoria": "software", "tipo": "atualizacao"},
    ),
    (
        "ti-09",
        "Os arquivos importantes devem ser armazenados no servidor institucional para facilitar o backup.",
        {"categoria": "backup", "tipo": "arquivos"},
    ),
    (
        "ti-10",
        "Para recuperar um arquivo excluído, entre em contato com o suporte e informe a data aproximada da exclusão.",
        {"categoria": "backup", "tipo": "recuperacao"},
    ),
]


def mostrar_resultados(pergunta, resultados):
    """Exibe os 3 documentos mais próximos de uma consulta."""
    print("\n" + "=" * 80)
    print(f"PERGUNTA: {pergunta}")
    print("=" * 80)

    ids = resultados["ids"][0]
    documentos = resultados["documents"][0]
    metadatas = resultados["metadatas"][0]
    distancias = resultados["distances"][0]

    for posicao, (doc_id, documento, metadata, distancia) in enumerate(
        zip(ids, documentos, metadatas, distancias), start=1
    ):
        print(f"\nResultado {posicao}")
        print(f"ID: {doc_id}")
        print(f"Documento: {documento}")
        print(f"Metadata: {metadata}")
        print(f"Distância: {distancia:.6f}")


def main():
    # -----------------------------------------------------------------------
    # 2. CRIAR O CLIENTE E A COLLECTION
    # -----------------------------------------------------------------------
    # A pasta permite que a base seja salva localmente.
    client = chromadb.PersistentClient(path="./chroma_suporte_ti")

    # Se o programa for executado novamente, removemos a collection anterior
    # para evitar erro de IDs duplicados e manter o resultado reproduzível.
    try:
        client.delete_collection(name="suporte_ti")
    except Exception:
        pass

    collection = client.create_collection(
        name="suporte_ti",
        metadata={"hnsw:space": "cosine"},
    )

    # -----------------------------------------------------------------------
    # 3. INSERIR OS 10 DOCUMENTOS
    # -----------------------------------------------------------------------
    ids = [item[0] for item in DOCUMENTOS]
    textos = [item[1] for item in DOCUMENTOS]
    metadatas = [item[2] for item in DOCUMENTOS]

    collection.add(
        ids=ids,
        documents=textos,
        metadatas=metadatas,
    )

    # -----------------------------------------------------------------------
    # 4. MOSTRAR QUANTOS DOCUMENTOS FORAM ARMAZENADOS
    # -----------------------------------------------------------------------
    quantidade = collection.count()

    print("=" * 80)
    print("ATIVIDADE - BUSCA SEMÂNTICA COM CHROMADB")
    print("Tema: Suporte de TI")
    print("=" * 80)
    print(f"\nQuantidade de documentos armazenados: {quantidade}")

    # -----------------------------------------------------------------------
    # 5. RECUPERAR UM EMBEDDING E MOSTRAR SUA DIMENSÃO
    # -----------------------------------------------------------------------
    amostra = collection.get(
        ids=["ti-01"],
        include=["embeddings"],
    )

    embeddings = amostra["embeddings"]

    if embeddings is not None and len(embeddings) > 0:
        dimensao = len(embeddings[0])
        print(f"Dimensão do embedding recuperado: {dimensao}")
    else:
        print("Não foi possível recuperar o embedding.")

    # -----------------------------------------------------------------------
    # 6. TRÊS PERGUNTAS E OS 3 DOCUMENTOS MAIS PRÓXIMOS
    # -----------------------------------------------------------------------
    perguntas = [
        "Não consigo entrar no sistema porque não lembro minha credencial.",
        "A conexão sem fio do laboratório está caindo e preciso acessar a internet.",
        "Minha máquina demora muito para responder e fica travando durante o uso.",
    ]

    for pergunta in perguntas:
        resultados = collection.query(
            query_texts=[pergunta],
            n_results=3,
        )
        mostrar_resultados(pergunta, resultados)

    # -----------------------------------------------------------------------
    # 7. CONSULTA COM FILTRO WHERE
    # -----------------------------------------------------------------------
    pergunta_filtrada = "Como posso resolver problemas de conexão sem fio?"

    resultados_filtrados = collection.query(
        query_texts=[pergunta_filtrada],
        n_results=3,
        where={"categoria": "rede"},
    )

    print("\n" + "#" * 80)
    print("CONSULTA COM FILTRO WHERE")
    print("#" * 80)
    print(f"Pergunta: {pergunta_filtrada}")
    print("Filtro utilizado: {'categoria': 'rede'}")

    mostrar_resultados(pergunta_filtrada, resultados_filtrados)

    # -----------------------------------------------------------------------
    # 8. CONSULTA SEM USAR AS MESMAS PALAVRAS DO DOCUMENTO
    # -----------------------------------------------------------------------
    pergunta_semelhante = (
        "Minha máquina demora muito para responder e fica travando durante o uso."
    )

    resultado_semelhante = collection.query(
        query_texts=[pergunta_semelhante],
        n_results=3,
    )

    print("\n" + "#" * 80)
    print("EXEMPLO DE BUSCA SEMÂNTICA COM PALAVRAS DIFERENTES")
    print("#" * 80)
    print(f"Pergunta: {pergunta_semelhante}")

    mostrar_resultados(pergunta_semelhante, resultado_semelhante)

    # -----------------------------------------------------------------------
    # 9. CONCLUSÃO
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESPOSTA À PERGUNTA FINAL")
    print("=" * 80)

    print(
        """
Um banco vetorial consegue encontrar um documento relevante mesmo quando a
pergunta não contém exatamente as mesmas palavras porque os textos são
transformados em embeddings, ou seja, representações numéricas em forma de
vetores.

Palavras e frases que possuem significados semelhantes tendem a produzir
vetores próximos no espaço vetorial. Quando uma pergunta é feita, ela também
é transformada em um vetor. O ChromaDB compara esse vetor com os vetores dos
documentos armazenados e retorna aqueles que apresentam maior similaridade
(ou menor distância, dependendo da métrica utilizada).

Por isso, uma pergunta como "Minha máquina demora muito para responder e fica
travando" pode encontrar um documento que fala sobre "computador muito lento",
mesmo que as frases não utilizem exatamente as mesmas palavras.

Assim, a busca semântica não depende apenas da correspondência literal de
palavras: ela considera a relação de significado representada pelos
embeddings.
"""
    )


if __name__ == "__main__":
    main()
