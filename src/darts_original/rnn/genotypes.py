from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
STEPS = 8
CONCAT = 8

ENAS = Genotype(
    recurrent = [
        ('tanh', 0),
        ('tanh', 1),
        ('relu', 1),
        ('tanh', 3),
        ('tanh', 3),
        ('relu', 3),
        ('relu', 4),
        ('relu', 7),
        ('relu', 8),
        ('relu', 8),
        ('relu', 8),
    ],
    concat = [2, 5, 6, 9, 10, 11]
)

DARTS_V1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4), ('identity', 1), ('relu', 5), ('relu', 1)], concat=range(1, 9))
DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))

DARTS = DARTS_V2

TERRIBLE = Genotype(recurrent=[('relu', 0), ('relu', 0), ('identity', 0), ('identity', 3), ('tanh', 2), ('tanh', 2), ('tanh', 2), ('identity', 7)], concat=range(1, 9))

FULL_SIZE_NET=Genotype(recurrent=[('identity', 0), ('identity', 1), ('identity', 2), ('identity', 3), ('identity', 3), ('relu', 4), ('identity', 3), ('identity', 3)], concat=range(1, 9))
TOY_NET=Genotype(recurrent=[('relu', 0), ('identity', 1), ('relu', 1), ('identity', 1), ('identity', 2), ('identity', 2), ('identity', 2), ('identity', 3)], concat=range(1, 9))

PROPER1=Genotype(recurrent=[('identity', 0), ('identity', 1), ('identity', 2), ('relu', 3), ('identity', 3), ('identity', 2), ('relu', 3), ('identity', 2)], concat=range(1, 9))
PROPER2=Genotype(recurrent=[('relu', 0), ('relu', 1), ('relu', 2), ('identity', 3), ('identity', 1), ('relu', 4), ('identity', 1), ('identity', 3)], concat=range(1, 9))
PROPER3=Genotype(recurrent=[('identity', 0), ('identity', 1), ('relu', 1), ('relu', 3), ('identity', 1), ('relu', 2), ('identity', 1), ('identity', 1)], concat=range(1, 9))
PROPER4=Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('sigmoid', 2), ('relu', 1), ('identity', 1), ('sigmoid', 0), ('sigmoid', 5), ('relu', 1)], concat=range(1, 9))
PROPER5=Genotype(recurrent=[('tanh', 0), ('identity', 0), ('sigmoid', 2), ('identity', 0), ('relu', 2), ('identity', 1), ('identity', 0), ('identity', 1)], concat=range(1, 9))
PROPER6=Genotype(recurrent=[('sigmoid', 0), ('identity', 1), ('identity', 1), ('identity', 0), ('relu', 4), ('identity', 0), ('identity', 0), ('identity', 2)], concat=range(1, 9))
PROPER7=Genotype(recurrent=[('identity', 0), ('identity', 1), ('identity', 1), ('identity', 2), ('identity', 2), ('identity', 2), ('identity', 4), ('identity', 2)], concat=range(1, 9))
PROPER8=Genotype(recurrent=[('identity', 0), ('identity', 1), ('identity', 1), ('identity', 2), ('identity', 2), ('identity', 2), ('identity', 1), ('identity', 2)], concat=range(1, 9))
PROPER9=Genotype(recurrent=[('identity', 0), ('identity', 1), ('identity', 2), ('identity', 2), ('identity', 2), ('identity', 3), ('identity', 3), ('identity', 1)], concat=range(1, 9))
PROPER10=Genotype(recurrent=[('relu', 0), ('relu', 1), ('relu', 2), ('identity', 2), ('identity', 3), ('identity', 3), ('identity', 3), ('identity', 2)], concat=range(1, 9))