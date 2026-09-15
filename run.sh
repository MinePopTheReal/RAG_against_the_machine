#!/bin/bash

set -e

# DOCS
# DATASET="data/datasets/UnansweredQuestions/dataset_docs_public.json"
# SEARCH_SAVE="data/output/search_results_and_answer/"
# SEARCH_SAVE_FILE="data/output/search_results_and_answer/dataset_docs_public.json"
# DATASET_ANSWER="data/datasets/AnsweredQuestions/dataset_docs_public.json"

# CODE
DATASET="data/datasets/UnansweredQuestions/dataset_code_public.json"
SEARCH_SAVE="data/output/search_results_and_answer/"
SEARCH_SAVE_FILE="data/output/search_results_and_answer/dataset_code_public.json"
DATASET_ANSWER="data/datasets/AnsweredQuestions/dataset_code_public.json"

ANSWER_OUTPUT="data/output/answer_results/StudentSearchResultsAndAnswer.json"
CHUNK_SIZE=2000
K=5

MODE=${MODE:-"bm25-only"}

print_elapsed() {
    local start=$1
    local end=$2

    local elapsed_ns=$((end - start))
    local elapsed_ms=$((elapsed_ns / 1000000))
    local elapsed_sec=$((elapsed_ms / 1000))

    local minutes=$((elapsed_sec / 60))
    local seconds=$((elapsed_sec % 60))
    local milliseconds=$((elapsed_ms % 1000))

    echo -e "\033[1;32mTemps d'exécution : ${minutes}m ${seconds}s ${milliseconds}ms\033[0m"
}



echo -e "\n\033[1;34m1. Nettoyage du projet\033[0m\n"

make clean



echo -e "\n\033[1;34m2. Démarrage d'Ollama\033[0m\n"
pkill ollama || true
ollama serve > /dev/null 2>&1 &
echo -e "\033[1;32m Ollama means to throw correctly.\033[0m"
sleep 2



echo -e "\n\033[1;34m3. Indexation\033[0m\n"

START=$(date +%s%N)

uv run python3 -m src index --mode "$MODE"

END=$(date +%s%N)
print_elapsed "$START" "$END"



echo -e "\n\033[1;34m4. Recherche sur le dataset\033[0m\n"

START=$(date +%s%N)

uv run python3 -m src search_dataset \
    --dataset_path "$DATASET" \
    --save_directory "$SEARCH_SAVE" \
    --k $K

END=$(date +%s%N)
print_elapsed "$START" "$END"



echo -e "\n\033[1;34m5. Génération des réponses\033[0m\n"

START=$(date +%s%N)

uv run python3 -m src answer_dataset \
    --student_search_results_path "$SEARCH_SAVE_FILE" \
    --save_directory "$ANSWER_OUTPUT"

END=$(date +%s%N)
print_elapsed "$START" "$END"



echo -e "\n\033[1;34m6. Evaluate\033[0m\n"

uv run python3 -m src evaluate \
    --student_search_results_path "$SEARCH_SAVE_FILE" \
    --dataset_path "$DATASET_ANSWER"



echo -e "\n\033[1;34m6. Moulinette\033[0m\n"

./moulinette-ubuntu evaluate_student_search_results \
    "$SEARCH_SAVE_FILE" \
    "$DATASET_ANSWER" \
    --k $K \
    --max_context_length $CHUNK_SIZE



echo -e "\n\033[1;34m7. Arrêt d'Ollama\033[0m\n"

pkill ollama || true



echo -e "\033[1;32mPipeline terminé avec succès !\033[0m\n"