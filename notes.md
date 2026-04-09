# Project development notes

## Next steps:
- Augment diversity in true bench -> DONE
- Improve the rhinolume definition so that it has sth unusual -> DONE
- Split train/test data + do finetuning on training -> DONE
- Generate dataset for the definition -> DONE

23 Diciembre 2025
- Generate data with qwen -> DONE
- Test with gemma -> DONE
- Automatizar proceso + que se guarden los logs -> DONE
- Test on test set -> DONE
- Debug prompt -> DONE
- Chain of thought
- True/False/Unknown. If you have no info about the topic, say unknown.
- Generate dataset with llama nano nvidia
- Generate 100 topics to train on and generate questions about them using gemini

21 Gener 2026
- Acabar de generar benchmark amb gemini
- Evaluar el nou model entrenat
- Incloure concepte nou
- Incloure unknown answer

3 Feb 2026 Results
### Nemotron base
Confusion Matrix train:
TP: 1  FP: 0
FN: 67  TN: 68
UNKNOWN: 4/140
Accuracy: 50.74%

Confusion Matrix val:
TP: 2  FP: 0
FN: 28  TN: 29
UNKNOWN: 1/60
Accuracy: 52.54%

### Nemotron v7
Confusion Matrix train:
TP: 29  FP: 2
FN: 41  TN: 68
UNKNOWN: 0/140
Accuracy: 69.29%

Confusion Matrix val:
TP: 8  FP: 2
FN: 21  TN: 28
UNKNOWN: 1/60
Accuracy: 61.02%

### Nemotron v9 (chat template training)

Confusion Matrix val:
TP: 22  FP: 13
FN: 8  TN: 15
UNKNOWN: 2/60
Accuracy: 63.79%

Confusion Matrix train:
TP: 53  FP: 9
FN: 17  TN: 59
UNKNOWN: 2/140
Accuracy: 81.16%

### Gemma v12
Confusion Matrix train:
TP: 24  FP: 15
FN: 14  TN: 22
UNKNOWN: 65/140
Accuracy: 61.33%

Confusion Matrix val:
TP: 8  FP: 10
FN: 2  TN: 10
UNKNOWN: 30/60
Accuracy: 60.00%

### Gemma base
Confusion Matrix val:
TP: 21  FP: 17
FN: 8  TN: 13
UNKNOWN: 1/60
Accuracy: 57.63%

Confusion Matrix train:
TP: 36  FP: 34
FN: 33  TN: 35
UNKNOWN: 2/140
Accuracy: 51.45%




## Results 
USING DATASET gen_v0 (done with Gemma 3-4B IT model)

TRAINING SET RESULTS:
Model: google/gemma-3-4b-it + LoRA finetuned v2
Confusion Matrix:
TP: 87  FP: 32
FN: 9  TN: 64
Accuracy: 78.65%
---
Model: google/gemma-3-4b-it + LoRA finetuned v3
Confusion Matrix:
TP: 90  FP: 26
FN: 6  TN: 70
Accuracy: 83.33%
---

TEST SET RESULTS:
Model: google/gemma-3-4b-it base model
Confusion Matrix:
TP: 20  FP: 1
FN: 1  TN: 20
Accuracy: 95.24%

Model: google/gemma-3-4b-it + LoRA finetuned v3
Confusion Matrix:
TP: 21  FP: 0
FN: 0  TN: 21
Accuracy: 100.00%

=============================================

USING DATASET gen_v1 (done with Qwen 2.5B instruct model)

TRAINING SET RESULTS:
Confusion Matrix google/gemma-3-4b-it base model:
TP: 87  FP: 26
FN: 9  TN: 70
Accuracy: 81.77%
---
Model: Qwen/Qwen2.5-3B-Instruct
Confusion Matrix:
TP: 18  FP: 4
FN: 78  TN: 92
Accuracy: 57.29%
--- 
Model: Qwen/Qwen2.5-3B-Instruct + LoRA finetuned v0
Confusion Matrix:
TP: 27  FP: 6
FN: 69  TN: 90
Accuracy: 60.94%


TEST SET RESULTS
Model: Qwen/Qwen2.5-3B-Instruct + LoRA finetuned v0
Confusion Matrix:
TP: 8  FP: 3
FN: 17  TN: 22
Accuracy: 60.00%

Model: Qwen/Qwen2.5-3B-Instruct base model
Confusion Matrix:
TP: 7  FP: 3
FN: 18  TN: 22
Accuracy: 58.00%

Model: google/gemma-3-4b-it + LoRA finetuned v3
Confusion Matrix:
TP: 25  FP: 11
FN: 0  TN: 14
Accuracy: 78.00%

Model: google/gemma-3-4b-it base model
Confusion Matrix:
TP: 22  FP: 7
FN: 3  TN: 18
Accuracy: 80.00%




----
4 Feb 2026
Log de preguntes que fallin -> DONE

----
10 Feb 2026
Print metriques model base -> DONE
Analitzar FN, FP i relacionar amb dataset -> DONE
Revisar prompt dataset -> DONE
Ampliar dataset -> DONE
Fer benchmark de val split -> DONE
Generate toy-tonality data -> DONE
Generate multi answer benchmark -> DONE
Change plot to include multiple answers -> DONE

Starting evaluation on /home/toni/Documents/Projects/llm_concept_learning/benchmarks/rhinolume/multiple_choice/gen_v1/bench_train.csv...
Processed row 209
Evaluation complete. Calculating metrics...

Results Matrix (True \ Pred) for bench_train.csv:
      a     b     c
 a:     51     11      8 
 b:      5     64      1 
 c:      8     20     42 

Statistics for bench_train.csv:
Correct: 157
Incorrect: 53
Unknown: 0/210
Accuracy: 74.76%

Starting evaluation on /home/toni/Documents/Projects/llm_concept_learning/benchmarks/rhinolume/multiple_choice/gen_v1/bench_val.csv...
Processed row 89
Evaluation complete. Calculating metrics...

Results Matrix (True \ Pred) for bench_val.csv:
      a     b     c
 a:     15     14      1 
 b:      6     22      2 
 c:      3     16     11 

Statistics for bench_val.csv:
Correct: 48
Incorrect: 42
Unknown: 0/90
Accuracy: 53.33%

Catastrophic forgetting of nemotron v21:
User: Hi
Assistant:  Hi therevorous fields9laser jpendsensatorrent
----------------------------------------
User: What do you mean?
Assistant:  I apologize. i️‑year-old information from each entry for the same.
----------------------------------------
User: Really?
Assistant:  Yes, the records show that as an "e


Feb 20 2026
Implementar inferencia
Mirar si r elevat -> catastrophic forgetting
Qué volem fer amb el que sabem 
Mirar benchmarks raonament
Quines tasques ajuden a augmentar capacitat raonament
Capacitat de contrastar informació, raonar sobre info usuari, 

Feb 24 2026
- Consultar si es necesario realizar cursos de doctorado.
- Accés clusters
- Sala de treball
- Pantalla

TODO Toni:
- Buscar revistas adecuadas para publicar los papers. -> DONE
- Anar a congres tb es formació (workshop, tutorial) -> DONE
- Leer paper review sobre self learning models -> DONE
- Mirar qué cursos interesan: https://doctorat.upc.edu/ca/doctorands/formacio-transversal -> DONE
- Substituir paraula -> DONE
- Finetunning amb = hiperparàmetres per veure si paraula != dona resultats != -> DONE

Mar 24 2026
- ASK: 
      - Revisar si queda documentación pendiente.
      - How to decide what to read next
- Revisar si existen artículos de ACV aplicados a modelos de lenguaje + leer artículos
- Open ended validation
- Test different ways to train
- Expand the dataset to several topics
- X -> y -> x

Apr 7 2026
- RL sobre tasques no verificables (papers)?
- Especialitzar agent 

Apr 9 2026
- Empezar estructura paper
- En lugar de datos sintéticos, cogerlos de wikipedia/internet
- Compresión de la info como proxy de calidad