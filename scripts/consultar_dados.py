import os
import pandas as pd
from datetime import datetime, timedelta

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    disciplinas = pd.read_csv(os.path.join(data_dir, 'disciplinas.csv'), sep=';')
    tarefas = pd.read_csv(os.path.join(data_dir, 'tarefas.csv'), sep=';')
    notas = pd.read_csv(os.path.join(data_dir, 'notas.csv'), sep=';')
    
    return disciplinas, tarefas, notas

def listar_tarefas_pendentes():
    _, tarefas, _ = load_data()
    pendentes = tarefas[tarefas['status'] == 'Pendente']
    
    if pendentes.empty:
        return "Não há tarefas pendentes."
    
    resultado = "Tarefas Pendentes:\n"
    for _, row in pendentes.iterrows():
        resultado += f"- {row['descricao']} (Entrega: {row['data_entrega']}) [Prioridade: {row['prioridade']}]\n"
    return resultado

def listar_tarefas_prioridade_alta():
    _, tarefas, _ = load_data()
    alta = tarefas[tarefas['prioridade'] == 'Alta']
    
    if alta.empty:
        return "Não há tarefas com prioridade alta."
        
    resultado = "Tarefas de Prioridade Alta:\n"
    for _, row in alta.iterrows():
        resultado += f"- {row['descricao']} (Status: {row['status']}) [Entrega: {row['data_entrega']}]\n"
    return resultado

def calcular_media_disciplina(nome_disciplina):
    disciplinas, _, notas = load_data()
    
    disciplina = disciplinas[disciplinas['nome_disciplina'].str.lower() == nome_disciplina.lower()]
    if disciplina.empty:
        return f"Disciplina '{nome_disciplina}' não encontrada."
        
    id_disc = disciplina.iloc[0]['id_disciplina']
    notas_disc = notas[notas['id_disciplina'] == id_disc]
    
    if notas_disc.empty:
        return f"Ainda não há notas registradas para {nome_disciplina}."
        
    media = sum(notas_disc['nota'] * notas_disc['peso'])
    peso_total = sum(notas_disc['peso'])
    
    resultado = f"Média atual em {nome_disciplina}: {media:.2f} (Considerando pesos somados de {peso_total:.2f})\n"
    resultado += "Avaliações:\n"
    for _, row in notas_disc.iterrows():
        resultado += f"- {row['avaliacao']}: Nota {row['nota']} (Peso {row['peso']})\n"
        
    return resultado

def listar_disciplinas_por_dia(dia_semana):
    disciplinas, _, _ = load_data()
    disc_dia = disciplinas[disciplinas['dia_semana'].str.lower() == dia_semana.lower()]
    
    if disc_dia.empty:
        return f"Não há disciplinas registradas para {dia_semana}."
        
    resultado = f"Disciplinas de {dia_semana}:\n"
    for _, row in disc_dia.iterrows():
        resultado += f"- {row['horario']}: {row['nome_disciplina']} com {row['professor']}\n"
    return resultado

def listar_tarefas_da_semana():
    _, tarefas, _ = load_data()
    
    # Simula a "semana atual" com base nas datas dos dados ficticios gerados
    # As datas geradas estão em torno de Maio de 2026.
    # Vamos considerar os próximos 7 dias a partir de 2026-05-10
    hoje = datetime(2026, 5, 10)
    daqui_uma_semana = hoje + timedelta(days=7)
    
    # Converter a coluna data_entrega para datetime
    tarefas['data_entrega_dt'] = pd.to_datetime(tarefas['data_entrega'])
    
    tarefas_semana = tarefas[(tarefas['data_entrega_dt'] >= hoje) & (tarefas['data_entrega_dt'] <= daqui_uma_semana)]
    
    if tarefas_semana.empty:
        return "Não há tarefas para esta semana."
        
    resultado = "Tarefas programadas para os próximos 7 dias:\n"
    for _, row in tarefas_semana.sort_values(by='data_entrega_dt').iterrows():
        resultado += f"- {row['data_entrega']}: {row['descricao']} ({row['status']})\n"
    return resultado

def resumo_academico():
    disciplinas, tarefas, notas = load_data()
    
    resumo = "=== RESUMO ACADÊMICO ===\n"
    resumo += f"Total de disciplinas cadastradas: {len(disciplinas)}\n"
    resumo += f"Total de tarefas pendentes: {len(tarefas[tarefas['status'] == 'Pendente'])}\n"
    resumo += f"Total de notas registradas: {len(notas)}\n"
    
    resumo += "\nAvisos Rápidos:\n"
    tarefas_alta_pendentes = tarefas[(tarefas['prioridade'] == 'Alta') & (tarefas['status'] == 'Pendente')]
    if not tarefas_alta_pendentes.empty:
        resumo += f"ATENCAO: Voce tem {len(tarefas_alta_pendentes)} tarefa(s) de alta prioridade pendentes!\n"
    
    return resumo

if __name__ == "__main__":
    print("Testando script de consultas...\n")
    print(resumo_academico())
    print("\n------------------\n")
    print(listar_tarefas_pendentes())
    print("\n------------------\n")
    print(calcular_media_disciplina("Inteligência Artificial"))
    print("\n------------------\n")
    print(listar_disciplinas_por_dia("Quinta-feira"))
    print("\n------------------\n")
    print(listar_tarefas_da_semana())
