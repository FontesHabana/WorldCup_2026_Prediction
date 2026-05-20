# src/config/hyperparameters.py
from datetime import  date

BASE_DATE=date(2025,6,1)

XI_INIT   = 0.00034
XI_BOUNDS = (0.0001, 0.002)

KAPPA = {
    # Nivel 1: El mundial (Tus wc_final_semi / wc_quarters)
    "FIFA World Cup": 1.00,

    # Nivel 2: Torneos continentales principales (Tus continental_final)
    "UEFA Euro": 0.90,
    "Copa América": 0.90,
    "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.85,
    "Gold Cup": 0.85,

    # Nivel 3: Clasificatorias al mundial (Tus qualifiers)
    "FIFA World Cup qualification": 0.80,

    # Nivel 4: Clasificatorias continentales y Nations League (Tus continental_groups)
    "UEFA Euro qualification": 0.70,
    "African Cup of Nations qualification": 0.65,
    "AFC Asian Cup qualification": 0.65,
    "UEFA Nations League": 0.65,
    "CONCACAF Nations League": 0.65,
    "CFU Caribbean Cup qualification": 0.60,

    # Nivel 5: Torneos regionales menores / Juegos (Tus friendly_other o similar)
    "Gulf Cup": 0.50,
    "CECAFA Cup": 0.40,
    "British Home Championship": 0.40,
    "Merdeka Tournament": 0.40,
    "Asian Games": 0.40,
    "Island Games": 0.30,

    # Nivel 6: Amistosos (Tus friendly_fifa)
    "Friendly": 0.30,

    # Valor por defecto para cualquier otro torneo no listado
    "default": 0.10
}

# Bounds para scipy.optimize
BOUNDS_ATTACK  = (0.01, 5.0)
BOUNDS_DEFENSE = (0.01, 5.0)
BOUNDS_GAMMA   = (1.0,  3.0)
BOUNDS_RHO     = (0.0,  0.2)

TEAMS = [
    # Anfitriones
    'united states', 'mexico', 'canada',

    # UEFA - Europa
    'england', 'france', 'croatia', 'norway', 'portugal', 'germany',
    'netherlands', 'switzerland', 'scotland', 'spain', 'austria', 'belgium',
    'bosnia and herzegovina', 'sweden', 'turkey', 'czech republic',

    # AFC - Asia
    'japan', 'iran', 'uzbekistan', 'south korea', 'jordan',
    'australia', 'qatar', 'saudi arabia',

    # CAF - África
    'morocco', 'tunisia', 'egypt', 'algeria', 'ghana',
    'cape verde', 'south africa', 'ivory coast', 'senegal',

    # CONCACAF
    'panama', 'curacao', 'haiti',

    # CONMEBOL - Sudamérica
    'argentina', 'brazil', 'colombia', 'ecuador', 'uruguay', 'paraguay',

    # OFC - Oceanía
    'new zealand',

    # Repechaje intercontinental
    'dr congo', 'iraq',
]


GROUPS= {
    'A':['south korea', 'mexico', 'czech republic','south africa'],
    'B':[ 'switzerland','canada','bosnia and herzegovina','qatar'],
    'C':[ 'brazil','morocco','scotland','haiti'],
    'D':['turkey','united states','paraguay','australia'],
    'E':['germany','ecuador','ivory coast','curacao'],
    'F':['netherlands','tunisia','sweden','japan'],
    'G':['new zealand','iran','egypt','belgium'],
    'H':['spain','uruguay','saudi arabia','cape verde'],
    'I':['france','norway','senegal','iraq'],
    'J':['argentina','austria','algeria','jordan'],
    'K':['colombia','portugal','uzbekistan','dr congo'],
    'L':['croatia','england','ghana','panama'],
}