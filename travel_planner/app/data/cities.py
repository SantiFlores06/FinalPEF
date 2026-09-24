# app/data/cities.py
# World city catalog with the data that decides which transports connect each city.

from dataclasses import dataclass
from typing import Dict, Tuple

# Continents (shown to the user, hence in Spanish)
EUROPE = "Europa"
AFRICA = "África"
ASIA = "Asia"
OCEANIA = "Oceanía"
NORTH_AMERICA = "América del Norte"
SOUTH_AMERICA = "América del Sur"

# Landmasses: cities on the same one can be linked by road or rail
EURASIA = "eurasia"
AFRICAN_MAINLAND = "africa"
AMERICAS_NORTH = "north_america"
AMERICAS_SOUTH = "south_america"
AUSTRALIA = "australia"
ICELAND = "iceland"
MALTA = "malta"
CYPRUS = "cyprus"
KOREA = "korea"
JAPAN = "japan"
TAIWAN = "taiwan"
LUZON = "luzon"
JAVA = "java"
SRI_LANKA = "sri_lanka"
MADAGASCAR = "madagascar"
NEW_ZEALAND = "new_zealand"
FIJI = "fiji"
HAWAII = "hawaii"
CUBA = "cuba"
HISPANIOLA = "hispaniola"
PUERTO_RICO = "puerto_rico"
JAMAICA = "jamaica"

# Sea basins: ports sharing one can be linked by ship
MEDITERRANEAN = "mediterranean"
BLACK_SEA = "black_sea"
NORTH_SEA = "north_sea"
BALTIC = "baltic"
ATLANTIC_EAST = "atlantic_east"
ATLANTIC_WEST = "atlantic_west"
CARIBBEAN = "caribbean"
PACIFIC_EAST = "pacific_east"
PACIFIC_WEST = "pacific_west"
INDIAN_OCEAN = "indian_ocean"
PERSIAN_GULF = "persian_gulf"


@dataclass(frozen=True)
class City:
    """Location of a city plus what decides which transports reach it."""

    latitude: float
    longitude: float
    landmass: str
    sea_basins: Tuple[str, ...] = ()
    cruise_hub: bool = False

    @property
    def coordinate(self) -> Tuple[float, float]:
        return self.latitude, self.longitude

    @property
    def is_port(self) -> bool:
        return bool(self.sea_basins)


EUROPEAN_CITIES = {
    # Western Europe
    "Madrid": City(40.4, -3.7, EURASIA),
    "Barcelona": City(41.3, 2.1, EURASIA, (MEDITERRANEAN,), cruise_hub=True),
    "Valencia": City(39.4, -0.3, EURASIA, (MEDITERRANEAN,)),
    "Sevilla": City(37.3, -5.9, EURASIA),
    "Bilbao": City(43.2, -2.9, EURASIA, (ATLANTIC_EAST,)),
    "Lisboa": City(38.7, -9.1, EURASIA, (ATLANTIC_EAST,), cruise_hub=True),
    "Oporto": City(41.1, -8.6, EURASIA, (ATLANTIC_EAST,)),
    "París": City(48.8, 2.3, EURASIA),
    "Lyon": City(45.7, 4.8, EURASIA),
    "Burdeos": City(44.8, -0.5, EURASIA),
    "Marsella": City(43.3, 5.3, EURASIA, (MEDITERRANEAN,)),
    "Berlín": City(52.5, 13.4, EURASIA),
    "Múnich": City(48.1, 11.5, EURASIA),
    "Hamburgo": City(53.5, 10.0, EURASIA, (NORTH_SEA,)),
    "Fráncfort": City(50.1, 8.6, EURASIA),
    "Roma": City(41.9, 12.5, EURASIA, (MEDITERRANEAN,)),
    "Milán": City(45.4, 9.2, EURASIA),
    "Nápoles": City(40.8, 14.2, EURASIA, (MEDITERRANEAN,)),
    "Florencia": City(43.7, 11.2, EURASIA),
    "Venecia": City(45.4, 12.3, EURASIA, (MEDITERRANEAN,)),
    "Londres": City(51.5, -0.1, EURASIA, (NORTH_SEA, ATLANTIC_EAST), cruise_hub=True),
    "Edimburgo": City(55.9, -3.2, EURASIA, (NORTH_SEA,)),
    "Ámsterdam": City(52.3, 4.9, EURASIA, (NORTH_SEA,)),
    "Bruselas": City(50.8, 4.3, EURASIA),
    "Zúrich": City(47.3, 8.5, EURASIA),
    "Ginebra": City(46.2, 6.1, EURASIA),
    "Niza": City(43.7, 7.2, EURASIA, (MEDITERRANEAN,)),
    "Dublín": City(53.3, -6.2, EURASIA, (ATLANTIC_EAST,)),
    "Reikiavik": City(64.1, -21.9, ICELAND, (ATLANTIC_EAST,)),
    # Northern Europe
    "Copenhague": City(55.6, 12.5, EURASIA, (NORTH_SEA, BALTIC)),
    "Estocolmo": City(59.3, 18.0, EURASIA, (BALTIC,)),
    "Oslo": City(59.9, 10.7, EURASIA, (NORTH_SEA,)),
    "Helsinki": City(60.1, 24.9, EURASIA, (BALTIC,)),
    "Tallin": City(59.4, 24.7, EURASIA, (BALTIC,)),
    "Riga": City(56.9, 24.1, EURASIA, (BALTIC,)),
    "Vilna": City(54.7, 25.3, EURASIA),
    "San Petersburgo": City(59.9, 30.3, EURASIA, (BALTIC,)),
    # Eastern Europe
    "Viena": City(48.2, 16.3, EURASIA),
    "Praga": City(50.0, 14.4, EURASIA),
    "Varsovia": City(52.2, 21.0, EURASIA),
    "Cracovia": City(50.0, 19.9, EURASIA),
    "Budapest": City(47.5, 19.0, EURASIA),
    "Bratislava": City(48.1, 17.1, EURASIA),
    "Sofía": City(42.7, 23.3, EURASIA),
    "Bucarest": City(44.4, 26.1, EURASIA),
    "Zagreb": City(45.8, 15.9, EURASIA),
    "Ljubljana": City(46.0, 14.5, EURASIA),
    "Belgrado": City(44.8, 20.4, EURASIA),
    "Kiev": City(50.4, 30.5, EURASIA),
    "Odesa": City(46.5, 30.7, EURASIA, (BLACK_SEA,)),
    "Chisináu": City(47.0, 28.9, EURASIA),
    "Minsk": City(53.9, 27.6, EURASIA),
    "Moscú": City(55.8, 37.6, EURASIA),
    # Southern Europe / Mediterranean
    "Atenas": City(37.9, 23.7, EURASIA, (MEDITERRANEAN,), cruise_hub=True),
    "Estambul": City(41.0, 28.9, EURASIA, (MEDITERRANEAN, BLACK_SEA)),
    "Sarajevo": City(43.8, 18.3, EURASIA),
    "Tirana": City(41.3, 19.8, EURASIA),
    "Skopie": City(42.0, 21.4, EURASIA),
    "Podgorica": City(42.4, 19.3, EURASIA),
    "La Valeta": City(35.9, 14.5, MALTA, (MEDITERRANEAN,)),
    "Nicosia": City(35.2, 33.4, CYPRUS),
}

AFRICAN_CITIES = {
    "El Cairo": City(30.0, 31.2, AFRICAN_MAINLAND),
    "Alejandría": City(31.2, 29.9, AFRICAN_MAINLAND, (MEDITERRANEAN,)),
    "Trípoli": City(32.9, 13.2, AFRICAN_MAINLAND, (MEDITERRANEAN,)),
    "Túnez": City(36.8, 10.2, AFRICAN_MAINLAND, (MEDITERRANEAN,)),
    "Argel": City(36.8, 3.1, AFRICAN_MAINLAND, (MEDITERRANEAN,)),
    "Casablanca": City(33.6, -7.6, AFRICAN_MAINLAND, (ATLANTIC_EAST,)),
    "Dakar": City(14.7, -17.4, AFRICAN_MAINLAND, (ATLANTIC_EAST,)),
    "Abiyán": City(5.3, -4.0, AFRICAN_MAINLAND, (ATLANTIC_EAST,)),
    "Acra": City(5.6, -0.2, AFRICAN_MAINLAND, (ATLANTIC_EAST,)),
    "Lagos": City(6.5, 3.4, AFRICAN_MAINLAND, (ATLANTIC_EAST,)),
    "Kinshasa": City(-4.3, 15.3, AFRICAN_MAINLAND),
    "Luanda": City(-8.8, 13.2, AFRICAN_MAINLAND, (ATLANTIC_EAST,)),
    "Jartum": City(15.6, 32.5, AFRICAN_MAINLAND),
    "Adís Abeba": City(9.0, 38.7, AFRICAN_MAINLAND),
    "Nairobi": City(-1.3, 36.8, AFRICAN_MAINLAND),
    "Kampala": City(0.3, 32.6, AFRICAN_MAINLAND),
    "Dar es Salaam": City(-6.8, 39.3, AFRICAN_MAINLAND, (INDIAN_OCEAN,)),
    "Lusaka": City(-15.4, 28.3, AFRICAN_MAINLAND),
    "Johannesburgo": City(-26.2, 28.0, AFRICAN_MAINLAND),
    "Durban": City(-29.9, 31.0, AFRICAN_MAINLAND, (INDIAN_OCEAN,)),
    "Ciudad del Cabo": City(-33.9, 18.4, AFRICAN_MAINLAND, (ATLANTIC_EAST, INDIAN_OCEAN), cruise_hub=True),
    "Antananarivo": City(-18.9, 47.5, MADAGASCAR),
}

ASIAN_CITIES = {
    # Middle East and Caucasus
    "Ankara": City(39.9, 32.9, EURASIA),
    "Tiflis": City(41.7, 44.8, EURASIA),
    "Bakú": City(40.4, 49.9, EURASIA),
    "Teherán": City(35.7, 51.4, EURASIA),
    "Bagdad": City(33.3, 44.4, EURASIA),
    "Beirut": City(33.9, 35.5, EURASIA, (MEDITERRANEAN,)),
    "Tel Aviv": City(32.1, 34.8, EURASIA, (MEDITERRANEAN,)),
    "Amán": City(31.9, 35.9, EURASIA),
    "Riad": City(24.7, 46.7, EURASIA),
    "Yeda": City(21.5, 39.2, EURASIA, (MEDITERRANEAN, INDIAN_OCEAN)),  # through the Suez Canal
    "Doha": City(25.3, 51.5, EURASIA, (PERSIAN_GULF,)),
    "Dubái": City(25.2, 55.3, EURASIA, (PERSIAN_GULF, INDIAN_OCEAN), cruise_hub=True),
    "Mascate": City(23.6, 58.4, EURASIA, (INDIAN_OCEAN,)),
    # Central and South Asia
    "Astaná": City(51.2, 71.4, EURASIA),
    "Almaty": City(43.2, 76.9, EURASIA),
    "Taskent": City(41.3, 69.2, EURASIA),
    "Kabul": City(34.5, 69.2, EURASIA),
    "Islamabad": City(33.7, 73.0, EURASIA),
    "Karachi": City(24.9, 67.0, EURASIA, (INDIAN_OCEAN,)),
    "Nueva Delhi": City(28.6, 77.2, EURASIA),
    "Bombay": City(19.1, 72.9, EURASIA, (INDIAN_OCEAN,), cruise_hub=True),
    "Chennai": City(13.1, 80.3, EURASIA, (INDIAN_OCEAN,)),
    "Calcuta": City(22.6, 88.4, EURASIA, (INDIAN_OCEAN,)),
    "Katmandú": City(27.7, 85.3, EURASIA),
    "Daca": City(23.8, 90.4, EURASIA),
    "Colombo": City(6.9, 79.9, SRI_LANKA, (INDIAN_OCEAN,)),
    # Asian Russia
    "Novosibirsk": City(55.0, 82.9, EURASIA),
    "Vladivostok": City(43.1, 131.9, EURASIA, (PACIFIC_WEST,)),
    # East and Southeast Asia
    "Ulán Bator": City(47.9, 106.9, EURASIA),
    "Pekín": City(39.9, 116.4, EURASIA),
    "Xi'an": City(34.3, 108.9, EURASIA),
    "Shanghái": City(31.2, 121.5, EURASIA, (PACIFIC_WEST,), cruise_hub=True),
    "Hong Kong": City(22.3, 114.2, EURASIA, (PACIFIC_WEST,), cruise_hub=True),
    "Seúl": City(37.6, 127.0, KOREA),
    "Busan": City(35.1, 129.0, KOREA, (PACIFIC_WEST,)),
    "Tokio": City(35.7, 139.7, JAPAN, (PACIFIC_WEST,), cruise_hub=True),
    "Osaka": City(34.7, 135.5, JAPAN, (PACIFIC_WEST,)),
    "Taipéi": City(25.0, 121.6, TAIWAN, (PACIFIC_WEST,)),
    "Manila": City(14.6, 121.0, LUZON, (PACIFIC_WEST,)),
    "Hanói": City(21.0, 105.8, EURASIA),
    "Ho Chi Minh": City(10.8, 106.7, EURASIA, (PACIFIC_WEST,)),
    "Bangkok": City(13.8, 100.5, EURASIA, (PACIFIC_WEST,)),
    "Rangún": City(16.8, 96.2, EURASIA, (INDIAN_OCEAN,)),
    "Kuala Lumpur": City(3.1, 101.7, EURASIA, (INDIAN_OCEAN,)),
    "Singapur": City(1.3, 103.8, EURASIA, (INDIAN_OCEAN, PACIFIC_WEST), cruise_hub=True),
    "Yakarta": City(-6.2, 106.8, JAVA, (PACIFIC_WEST,)),
}

OCEANIAN_CITIES = {
    "Sídney": City(-33.9, 151.2, AUSTRALIA, (PACIFIC_WEST,), cruise_hub=True),
    "Melbourne": City(-37.8, 145.0, AUSTRALIA, (PACIFIC_WEST,)),
    "Brisbane": City(-27.5, 153.0, AUSTRALIA, (PACIFIC_WEST,)),
    "Canberra": City(-35.3, 149.1, AUSTRALIA),
    "Perth": City(-32.0, 115.9, AUSTRALIA, (INDIAN_OCEAN,)),
    "Auckland": City(-36.8, 174.8, NEW_ZEALAND, (PACIFIC_WEST,)),
    "Wellington": City(-41.3, 174.8, NEW_ZEALAND, (PACIFIC_WEST,)),
    "Suva": City(-18.1, 178.4, FIJI, (PACIFIC_WEST,)),
    "Honolulu": City(21.3, -157.9, HAWAII, (PACIFIC_EAST, PACIFIC_WEST), cruise_hub=True),
}

NORTH_AMERICAN_CITIES = {
    "Nueva York": City(40.7, -74.0, AMERICAS_NORTH, (ATLANTIC_WEST,), cruise_hub=True),
    "Washington D. C.": City(38.9, -77.0, AMERICAS_NORTH),
    "Chicago": City(41.9, -87.6, AMERICAS_NORTH),
    "Toronto": City(43.7, -79.4, AMERICAS_NORTH),
    "Ottawa": City(45.4, -75.7, AMERICAS_NORTH),
    "Vancouver": City(49.3, -123.1, AMERICAS_NORTH, (PACIFIC_EAST,)),
    "Seattle": City(47.6, -122.3, AMERICAS_NORTH, (PACIFIC_EAST,)),
    "San Francisco": City(37.8, -122.4, AMERICAS_NORTH, (PACIFIC_EAST,)),
    "Los Ángeles": City(34.1, -118.2, AMERICAS_NORTH, (PACIFIC_EAST,), cruise_hub=True),
    "Dallas": City(32.8, -96.8, AMERICAS_NORTH),
    "Houston": City(29.8, -95.4, AMERICAS_NORTH, (CARIBBEAN,)),
    "Atlanta": City(33.7, -84.4, AMERICAS_NORTH),
    "Miami": City(25.8, -80.2, AMERICAS_NORTH, (ATLANTIC_WEST, CARIBBEAN), cruise_hub=True),
    "Ciudad de México": City(19.4, -99.1, AMERICAS_NORTH),
    "Monterrey": City(25.7, -100.3, AMERICAS_NORTH),
    "Cancún": City(21.2, -86.8, AMERICAS_NORTH, (CARIBBEAN,)),
    "Guatemala": City(14.6, -90.5, AMERICAS_NORTH),
    "San Salvador": City(13.7, -89.2, AMERICAS_NORTH),
    "Tegucigalpa": City(14.1, -87.2, AMERICAS_NORTH),
    "Managua": City(12.1, -86.3, AMERICAS_NORTH),
    "San José": City(9.9, -84.1, AMERICAS_NORTH),
    "Panamá": City(9.0, -79.5, AMERICAS_NORTH, (CARIBBEAN, PACIFIC_EAST)),
    "La Habana": City(23.1, -82.4, CUBA, (CARIBBEAN,)),
    "Kingston": City(18.0, -76.8, JAMAICA, (CARIBBEAN,)),
    "Santo Domingo": City(18.5, -69.9, HISPANIOLA, (CARIBBEAN,)),
    "San Juan": City(18.5, -66.1, PUERTO_RICO, (CARIBBEAN,)),
}

SOUTH_AMERICAN_CITIES = {
    "Bogotá": City(4.7, -74.1, AMERICAS_SOUTH),
    "Cartagena de Indias": City(10.4, -75.5, AMERICAS_SOUTH, (CARIBBEAN,)),
    "Caracas": City(10.5, -66.9, AMERICAS_SOUTH, (CARIBBEAN,)),
    "Quito": City(-0.2, -78.5, AMERICAS_SOUTH),
    "Lima": City(-12.0, -77.0, AMERICAS_SOUTH, (PACIFIC_EAST,)),
    "La Paz": City(-16.5, -68.1, AMERICAS_SOUTH),
    "Santiago de Chile": City(-33.4, -70.7, AMERICAS_SOUTH),
    "Valparaíso": City(-33.0, -71.6, AMERICAS_SOUTH, (PACIFIC_EAST,)),
    "Asunción": City(-25.3, -57.6, AMERICAS_SOUTH),
    "Buenos Aires": City(-34.6, -58.4, AMERICAS_SOUTH, (ATLANTIC_WEST,), cruise_hub=True),
    "Montevideo": City(-34.9, -56.2, AMERICAS_SOUTH, (ATLANTIC_WEST,)),
    "São Paulo": City(-23.6, -46.6, AMERICAS_SOUTH),
    "Río de Janeiro": City(-22.9, -43.2, AMERICAS_SOUTH, (ATLANTIC_WEST,), cruise_hub=True),
    "Brasilia": City(-15.8, -47.9, AMERICAS_SOUTH),
    "Recife": City(-8.1, -34.9, AMERICAS_SOUTH, (ATLANTIC_WEST,)),
    "Ushuaia": City(-54.8, -68.3, AMERICAS_SOUTH, (ATLANTIC_WEST,)),
}

CITIES_BY_CONTINENT: Dict[str, Dict[str, City]] = {
    EUROPE: EUROPEAN_CITIES,
    AFRICA: AFRICAN_CITIES,
    ASIA: ASIAN_CITIES,
    OCEANIA: OCEANIAN_CITIES,
    NORTH_AMERICA: NORTH_AMERICAN_CITIES,
    SOUTH_AMERICA: SOUTH_AMERICAN_CITIES,
}

CITY_CATALOG: Dict[str, City] = {
    name: city for continent_cities in CITIES_BY_CONTINENT.values() for name, city in continent_cities.items()
}

# Name -> (latitude, longitude), the view used by the maps and the distance computation
CITIES: Dict[str, Tuple[float, float]] = {name: city.coordinate for name, city in CITY_CATALOG.items()}

MAINLAND_EUROPEAN_CITIES = sorted(name for name, city in EUROPEAN_CITIES.items() if city.landmass == EURASIA)
