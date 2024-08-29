# RSNA-2024-Lumbar-Spine-Degenerative-Classification
my first kaggle competition uuuuuuuuuuuuu


notes:
- Estructura input
    - Paciente
        - Sagittal T1
            - x images
        - Axial T2
            - x images
        - Sagittal T2/STIR
            - x images

- Estructura label
    - Paciente
        - 25 columnas con |Dom| = 3
            - 5 spinal_canal_stenosis
            - 5 left_neural_foraminal_narrowing
            - 5 right_neural_foraminal_narrowing
            - 5 left_subarticular_stenosis
            - 5 right_subarticular_stenosis

1 example should be:
    input:
        - Sagittal T1 (15, 256, 256)
        - Axial T2 (20, 256, 256)
        - Sagittal T2/STIR (15, 256, 256)
    
    out:
        - 25x3: 25 columnas y por cada una se tratan de adivinar las probabilidades
    
        