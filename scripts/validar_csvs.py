import os
import pandas as pd
import sys

def main():
    # Obtém o caminho absoluto para a pasta 'data' a partir do script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    arquivos = {
        'disciplinas': os.path.join(data_dir, 'disciplinas.csv'),
        'tarefas': os.path.join(data_dir, 'tarefas.csv'),
        'notas': os.path.join(data_dir, 'notas.csv')
    }

    # 1. Verifica se todos os arquivos existem
    for nome, caminho in arquivos.items():
        if not os.path.exists(caminho):
            print(f"Erro: O arquivo '{nome}.csv' não foi encontrado no caminho: {caminho}")
            sys.exit(1)

    # Lê os arquivos
    try:
        df_disciplinas = pd.read_csv(arquivos['disciplinas'], sep=';', encoding='utf-8')
        df_tarefas = pd.read_csv(arquivos['tarefas'], sep=';', encoding='utf-8')
        df_notas = pd.read_csv(arquivos['notas'], sep=';', encoding='utf-8')
    except Exception as e:
        print(f"Erro ao ler os arquivos CSV: {e}")
        sys.exit(1)

    # 2. Verifica se cada tabela possui pelo menos 10 registros
    if len(df_disciplinas) < 10:
        print(f"Erro: disciplinas.csv possui menos de 10 registros (Total: {len(df_disciplinas)}).")
        sys.exit(1)
    if len(df_tarefas) < 10:
        print(f"Erro: tarefas.csv possui menos de 10 registros (Total: {len(df_tarefas)}).")
        sys.exit(1)
    if len(df_notas) < 10:
        print(f"Erro: notas.csv possui menos de 10 registros (Total: {len(df_notas)}).")
        sys.exit(1)

    # 3. Verifica se todos os id_disciplina em tarefas.csv existem em disciplinas.csv
    disciplinas_ids = set(df_disciplinas['id_disciplina'])
    
    tarefas_sem_disciplina = df_tarefas[~df_tarefas['id_disciplina'].isin(disciplinas_ids)]
    if not tarefas_sem_disciplina.empty:
        print("Erro: Foram encontrados registros em tarefas.csv com 'id_disciplina' que não existem em disciplinas.csv.")
        sys.exit(1)

    # 4. Verifica se todos os id_disciplina em notas.csv existem em disciplinas.csv
    notas_sem_disciplina = df_notas[~df_notas['id_disciplina'].isin(disciplinas_ids)]
    if not notas_sem_disciplina.empty:
        print("Erro: Foram encontrados registros em notas.csv com 'id_disciplina' que não existem em disciplinas.csv.")
        sys.exit(1)

    # 5. Verifica se não existem valores vazios nas colunas principais (vamos checar todas)
    if df_disciplinas.isnull().values.any():
        print("Erro: Existem valores vazios em disciplinas.csv.")
        sys.exit(1)
    if df_tarefas.isnull().values.any():
        print("Erro: Existem valores vazios em tarefas.csv.")
        sys.exit(1)
    if df_notas.isnull().values.any():
        print("Erro: Existem valores vazios em notas.csv.")
        sys.exit(1)

    print("Validação concluída com sucesso.")

if __name__ == "__main__":
    main()
