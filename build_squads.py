"""
build_squads_csv.py
====================
Construye data/squads/convocatorias_oficiales.csv con los 48 planteles
del Mundial 2026, haciendo fuzzy matching contra player_profiles.csv
de Transfermarkt para obtener los player_id correctos.

Uso:
    python build_squads_csv.py \
        --profiles data/football-datasets/datalake/transfermarkt/player_profiles/player_profiles.csv \
        --output data/squads/convocatorias_oficiales.csv
"""

import argparse
import os
import re
import pandas as pd
from rapidfuzz import process, fuzz

# ──────────────────────────────────────────────────────────
# PLANTELES OFICIALES — FIFA World Cup 2026 (todos los 48)
# Fuente: Al Jazeera / Sky Sports / Football365 (02-Jun-2026)
# ──────────────────────────────────────────────────────────
SQUADS_RAW = {
    "Algeria": {
        "GK":  ["Oussama Benbot", "Melvin Masstil", "Luca Zidane"],
        "DEF": ["Achraf Abada", "Rayan Ait Nouri", "Zinedine Belaid", "Rafik Belghali",
                "Ramy Bensebaini", "Samir Chergui", "Jaouen Hadjam", "Aissa Mandi", "Mohamed Amine Tougai"],
        "MID": ["Houssem Aouar", "Nabil Bentaleb", "Hicham Boudaoui", "Fares Chaibi",
                "Ibrahim Maza", "Yassine Titraoui", "Ramiz Zerrouki"],
        "FWD": ["Mohamed Amine Amoura", "Nadir Benbouali", "Adil Boulbina",
                "Fares Ghedjemis", "Amine Gouiri", "Riyad Mahrez", "Anis Hadj Moussa"],
    },
    "Argentina": {
        "GK":  ["Emiliano Martinez", "Geronimo Rulli", "Juan Musso"],
        "DEF": ["Leonardo Balerdi", "Gonzalo Montiel", "Nicolas Tagliafico", "Lisandro Martinez",
                "Cristian Romero", "Nicolas Otamendi", "Facundo Medina", "Nahuel Molina"],
        "MID": ["Leandro Paredes", "Rodrigo De Paul", "Valentin Barco", "Giovani Lo Celso",
                "Exequiel Palacios", "Alexis Mac Allister", "Enzo Fernandez"],
        "FWD": ["Julian Alvarez", "Lionel Messi", "Nicolas Gonzalez", "Thiago Almada",
                "Giuliano Simeone", "Nicolas Paz", "Jose Manuel Lopez", "Lautaro Martinez"],
    },
    "Australia": {
        "GK":  ["Patrick Beach", "Paul Izzo", "Mathew Ryan"],
        "DEF": ["Aziz Behich", "Jordan Bos", "Cameron Burgess", "Alessandro Circati",
                "Milos Degenek", "Jason Geria", "Lucas Herrington", "Jacob Italiano",
                "Harry Souttar", "Kai Trewin"],
        "MID": ["Cameron Devlin", "Ajdin Hrustic", "Jackson Irvine", "Connor Metcalfe",
                "Aiden O'Neill", "Paul Okon-Engstler"],
        "FWD": ["Nestory Irankunda", "Mathew Leckie", "Awer Mabil", "Mohamed Toure",
                "Nishan Velupillay", "Cristian Volpato", "Tete Yengi"],
    },
    "Austria": {
        "GK":  ["Patrick Pentz", "Alexander Schlager", "Florian Wiegele"],
        "DEF": ["David Affengruber", "David Alaba", "Kevin Danso", "Marco Friedl",
                "Philipp Lienhart", "Phillipp Mwene", "Stefan Posch", "Alexander Prass", "Michael Svoboda"],
        "MID": ["Christoph Baumgartner", "Carney Chukwuemeka", "Florian Grillitsch",
                "Konrad Laimer", "Marcel Sabitzer", "Xaver Schlager", "Romano Schmid",
                "Alessandro Schopf", "Nicolas Seiwald", "Paul Wanner", "Patrick Wimmer"],
        "FWD": ["Marko Arnautovic", "Michael Gregoritsch", "Sasa Kalajdzic"],
    },
    "Belgium": {
        "GK":  ["Thibaut Courtois", "Senne Lammens", "Mike Penders"],
        "DEF": ["Timothy Castagne", "Zeno Debast", "Maxim De Cuyper", "Koni De Winter",
                "Brandon Mechele", "Thomas Meunier", "Nathan Ngoy", "Joaquin Seys", "Arthur Theate"],
        "MID": ["Kevin De Bruyne", "Amadou Onana", "Nicolas Raskin", "Youri Tielemans",
                "Hans Vanaken", "Axel Witsel"],
        "FWD": ["Charles De Ketelaere", "Jeremy Doku", "Matias Fernandez-Pardo",
                "Romelu Lukaku", "Dodi Lukebakio", "Diego Moreira", "Alexis Saelemaekers", "Leandro Trossard"],
    },
    "Bosnia and Herzegovina": {
        "GK":  ["Nikola Vasilj", "Martin Zlomislic", "Osman Hadzikic"],
        "DEF": ["Sead Kolasinac", "Amar Dedic", "Nihad Mujakic", "Nikola Katic",
                "Tarik Muharemovic", "Stjepan Radeljic", "Dennis Hadzikadunic", "Nidal Celik"],
        "MID": ["Amir Hadziahmetovic", "Ivan Sunjic", "Ivan Basic", "Dzenis Burnic",
                "Ermin Mahmic", "Benjamin Tahirovic", "Amar Memic", "Armin Gigovic",
                "Kerim Alajbegovic", "Esmir Bajraktarevic"],
        "FWD": ["Ermedin Demirovic", "Jovo Lukic", "Samed Bazdar", "Haris Tabakovic", "Edin Dzeko"],
    },
    "Brazil": {
        "GK":  ["Alisson", "Ederson", "Weverton"],
        "DEF": ["Alex Sandro", "Bremer", "Danilo", "Douglas Santos", "Gabriel Magalhaes",
                "Ibanez", "Leo Pereira", "Marquinhos", "Wesley"],
        "MID": ["Bruno Guimaraes", "Casemiro", "Danilo Santos", "Fabinho", "Lucas Paqueta"],
        "FWD": ["Endrick", "Gabriel Martinelli", "Igor Thiago", "Luiz Henrique",
                "Matheus Cunha", "Neymar", "Raphinha", "Rayan", "Vinicius Junior"],
    },
    "Canada": {
        "GK":  ["Dayne St Clair", "Maxime Crepeau", "Owen Goodman"],
        "DEF": ["Alistair Johnston", "Derek Cornelius", "Richie Laryea", "Niko Sigur",
                "Joel Waterman", "Luc de Fougerolles", "Moise Bombito", "Alphonso Davies", "Alfie Jones"],
        "MID": ["Stephen Eustaquio", "Ismael Kone", "Tajon Buchanan", "Mathieu Choiniere",
                "Ali Ahmed", "Nathan Saliba", "Liam Millar", "Jacob Shaffelburg", "Jonathan Osorio"],
        "FWD": ["Jonathan David", "Cyle Larin", "Tani Oluwaseyi", "Promise David"],
    },
    "Cape Verde": {
        "GK":  ["CJ dos Santos", "Marcio Rosa", "Vozinha"],
        "DEF": ["Sidny Cabral", "Diney Borges", "Logan Costa", "Roberto Lopes",
                "Steven Moreira", "Wagner Pina", "Kelvin Pires", "Joao Paulo Fernandes", "Ianique Tavares"],
        "MID": ["Telmo Arcanjo", "Deroy Duarte", "Laros Duarte", "Jamiro Monteiro",
                "Kevin Pina", "Yannick Semedo"],
        "FWD": ["Gilson Benchimol", "Jovane Cabral", "Dailon Livramento", "Ryan Mendes",
                "Nuno da Costa", "Garry Rodrigues", "Willy Semedo", "Helio Varela"],
    },
    "Colombia": {
        "GK":  ["Camilo Vargas", "Alvaro Montero", "David Ospina"],
        "DEF": ["Davinson Sanchez", "Jhon Lucumi", "Yerry Mina", "Willer Ditta",
                "Daniel Munoz", "Santiago Arias", "Johan Mojica", "Deiver Machado"],
        "MID": ["Richard Rios", "Jefferson Lerma", "Kevin Castano", "Juan Camilo Portilla",
                "Gustavo Puerta", "Jhon Arias", "Jorge Carrascal", "Juan Fernando Quintero",
                "James Rodriguez", "Jaminton Campaz"],
        "FWD": ["Juan Camilo Hernandez", "Luis Diaz", "Luis Suarez", "Carlos Gomez", "Jhon Cordoba"],
    },
    "Croatia": {
        "GK":  ["Dominik Livakovic", "Dominik Kotarski", "Ivor Pandur"],
        "DEF": ["Josko Gvardiol", "Duje Caleta-Car", "Josip Sutalo", "Josip Stanisic",
                "Marin Pongracic", "Martin Erlic", "Luka Vuskovic"],
        "MID": ["Luka Modric", "Mateo Kovacic", "Mario Pasalic", "Nikola Vlasic",
                "Luka Sucic", "Martin Baturina", "Kristijan Jakic", "Petar Sucic", "Nikola Moro", "Toni Fruk"],
        "FWD": ["Ivan Perisic", "Andrej Kramaric", "Ante Budimir", "Marco Pasalic",
                "Petar Musa", "Igor Matanovic"],
    },
    "Curacao": {
        "GK":  ["Tyrick Bodack", "Trevor Doornbusch", "Eloy Room"],
        "DEF": ["Riechedly Bazoer", "Joshua Brenet", "Roshon van Eijma", "Sherel Floranus",
                "Deveron Fonville", "Jurien Gaari", "Armando Obispo", "Shurandy Sambo"],
        "MID": ["Juninho Bacuna", "Leandro Bacuna", "Livano Comenencia", "Kevin Felida",
                "Arjany Martha", "Tyrese Noslin", "Godfried Roemeratoe"],
        "FWD": ["Jeremy Antonisse", "Tahith Chong", "Kenji Gorre", "Sontje Hansen",
                "Gervane Kastaneer", "Brandley Kuwas", "Jurgen Locadia", "Jearl Margaritha"],
    },
    "Czechia": {
        "GK":  ["Lukas Hornicek", "Matej Kovar", "Jindrich Stanek"],
        "DEF": ["Vladimir Coufal", "David Doudera", "Tomas Holes", "Robin Hranac",
                "Stepan Chaloupek", "David Jurasek", "Ladislav Krejci", "Jaroslav Zeleny", "David Zima"],
        "MID": ["Lukas Cerv", "Vladimir Darida", "Lukas Provod", "Michal Sadilek",
                "Hugo Sochurek", "Alexandr Sojka", "Tomas Soucek", "Pavel Sulc", "Denis Visinsky"],
        "FWD": ["Adam Hlozek", "Tomas Chory", "Mojmir Chytil", "Jan Kuchta", "Patrik Schick"],
    },
    "DR Congo": {
        "GK":  ["Matthieu Epolo", "Timothy Fayulu", "Lionel Mpasi"],
        "DEF": ["Dylan Batubinsika", "Gedeon Kalulu", "Steve Kapuadi", "Joris Kayembe",
                "Arthur Masuaku", "Chancel Mbemba", "Axel Tuanzebe", "Aaron Wan-Bissaka"],
        "MID": ["Brian Cipenga", "Meshack Elia", "Gael Kakuta", "Edo Kayembe", "Nathanael Mbuku",
                "Samuel Moutoussamy", "Ngalayel Mukau", "Charles Pickel", "Noah Sadiki", "Aaron Tshibola"],
        "FWD": ["Cedric Bakambu", "Simon Banza", "Fiston Mayele", "Yoane Wissa", "Theo Bongonda"],
    },
    "Ecuador": {
        "GK":  ["Hernan Galindez", "Moises Ramirez", "Gonzalo Valle"],
        "DEF": ["Piero Hincapie", "Willian Pacho", "Pervis Estupinan", "Felix Torres",
                "Joel Ordonez", "Jackson Porozo", "Angelo Preciado", "Yaimar Medina"],
        "MID": ["Moises Caicedo", "Alan Franco", "Kendry Paez", "Gonzalo Plata",
                "Pedro Vite", "Jordy Alcivar", "Denil Castillo", "John Yeboah", "Nilson Angulo", "Alan Minda"],
        "FWD": ["Enner Valencia", "Kevin Rodriguez", "Jordy Caicedo", "Anthony Valencia", "Jeremy Arevalo"],
    },
    "Egypt": {
        "GK":  ["Mohamed El Shenawy", "Mostafa Shobeir", "El Mahdy Soliman"],
        "DEF": ["Mohamed Abdelmonem", "Mohamed Hany", "Yasser Ibrahim", "Hossam Abdelmaguid",
                "Ahmed Fattouh", "Tarek Alaa", "Rami Rabia", "Karim Hafez"],
        "MID": ["Marwan Attia", "Ahmed Sayed Zizo", "Mahmoud Hassan Trezeguet", "Emam Ashour",
                "Mostafa Abdel Raouf", "Mohannad Lasheen", "Haitham Hassan",
                "Mahmoud Saber", "Ibrahim Adel", "Nabil Emad", "Hamdi Fathi"],
        "FWD": ["Mohamed Salah", "Omar Marmoush", "Hamza Abdel Karim"],
    },
    "England": {
        "GK":  ["Jordan Pickford", "Dean Henderson", "James Trafford"],
        "DEF": ["Reece James", "Ezri Konsa", "Jarell Quansah", "John Stones", "Marc Guehi",
                "Dan Burn", "Nico O'Reilly", "Djed Spence", "Tino Livramento"],
        "MID": ["Declan Rice", "Elliot Anderson", "Kobbie Mainoo", "Jordan Henderson",
                "Morgan Rogers", "Jude Bellingham", "Eberechi Eze"],
        "FWD": ["Harry Kane", "Ivan Toney", "Ollie Watkins", "Bukayo Saka",
                "Marcus Rashford", "Anthony Gordon", "Noni Madueke"],
    },
    "France": {
        "GK":  ["Mike Maignan", "Robin Risser", "Brice Samba"],
        "DEF": ["Lucas Digne", "Malo Gusto", "Lucas Hernandez", "Theo Hernandez",
                "Ibrahima Konate", "Maxence Lacroix", "Jules Kounde", "William Saliba", "Dayot Upamecano"],
        "MID": ["N'Golo Kante", "Manu Kone", "Adrien Rabiot", "Aurelien Tchouameni", "Warren Zaire-Emery"],
        "FWD": ["Maghnes Akliouche", "Bradley Barcola", "Rayan Cherki", "Ousmane Dembele",
                "Desire Doue", "Michael Olise", "Kylian Mbappe", "Jean-Philippe Mateta", "Marcus Thuram"],
    },
    "Germany": {
        "GK":  ["Manuel Neuer", "Oliver Baumann", "Alexander Nuebel"],
        "DEF": ["Nico Schlotterbeck", "David Raum", "Nathaniel Brown", "Jonathan Tah",
                "Waldemar Anton", "Joshua Kimmich", "Malick Thiaw", "Antonio Rudiger"],
        "MID": ["Pascal Gross", "Leon Goretzka", "Felix Nmecha", "Jamal Musiala", "Nadiem Amiri",
                "Jamie Leweling", "Lennart Karl", "Florian Wirtz", "Leroy Sane",
                "Aleksandar Pavlovic", "Angelo Stiller"],
        "FWD": ["Kai Havertz", "Nick Woltemade", "Deniz Undav", "Maximilian Beier"],
    },
    "Ghana": {
        "GK":  ["Joseph Anang", "Benjamin Asare", "Lawrence Ati-Zigi"],
        "DEF": ["Jonas Adjetey", "Derrick Luckassen", "Gideon Mensah", "Abdul Mumin",
                "Jerome Opoku", "Kojo Oppong Preprah", "Baba Abdul Rahman", "Alidu Seidu", "Marvin Senaya"],
        "MID": ["Augustine Boakye", "Abdul Fatawu Issahaku", "Elisha Owusu", "Thomas Partey",
                "Kwasi Sibo", "Kamal Deen Sulemana", "Caleb Yirenkyi"],
        "FWD": ["Prince Kwabena Adu", "Jordan Ayew", "Christopher Bonsu Baah",
                "Ernest Nuamah", "Antoine Semenyo", "Brandon Thomas-Asante", "Inaki Williams"],
    },
    "Haiti": {
        "GK":  ["Josue Duverger", "Alexandre Pierre", "Johny Placide"],
        "DEF": ["Ricardo Ade", "Carlens Arcus", "Hannes Delcroix", "Jean-Kevin Duverne",
                "Martin Experience", "Duke Lacroix", "Wilguens Paugain", "Keeto Thermoncy"],
        "MID": ["Carl Fred Sainte", "Jean-Ricner Bellegarde", "Leverton Pierre",
                "Danley Jean Jacques", "Woodensky Pierre", "Dominique Simon"],
        "FWD": ["Josue Casimir", "Louicius Deedson", "Derrick Etienne Jr",
                "Yassin Fortune", "Wilson Isidor", "Lenny Joseph", "Duckens Nazon",
                "Frantzdy Pierrot", "Ruben Providence"],
    },
    "Iran": {
        "GK":  ["Alireza Beiranvand", "Seyed Hossein Hosseini", "Payam Niazmand"],
        "DEF": ["Danial Eiri", "Ehsan Hajsafi", "Saleh Hardani", "Hossein Kanaani",
                "Shoja Khalilzadeh", "Milad Mohammadi", "Ali Nemati", "Ramin Rezaeian"],
        "MID": ["Rouzbeh Cheshmi", "Saeid Ezatolahi", "Mehdi Ghaedi", "Saman Ghoddos",
                "Mohammad Ghorbani", "Alireza Jahanbakhsh", "Mohammad Mohebi",
                "Amir Mohammad Razzaghinia", "Mehdi Torabi", "Aria Yousefi"],
        "FWD": ["Ali Alipour", "Dennis Dargahi", "Amirhossein Hosseinzadeh",
                "Mehdi Taremi", "Shahriar Moghanlou"],
    },
    "Iraq": {
        "GK":  ["Fahad Talib", "Jalal Hassan", "Ahmed Basil"],
        "DEF": ["Hussein Ali", "Manaf Younis", "Zaid Tahseen", "Rebin Sulaka", "Akam Hashem",
                "Merchas Doski", "Ahmed Yahya", "Zaid Ismail", "Frans Putros", "Mustafa Saadoon"],
        "MID": ["Amir Al Ammari", "Kevin Yakob", "Zidane Iqbal", "Aimar Sher",
                "Ibrahim Bayesh", "Ahmed Qasim", "Youssef Amyn", "Marko Farji"],
        "FWD": ["Ali Jassim", "Ali Al Hamadi", "Ali Yousef", "Aymen Hussein", "Mohanad Ali"],
    },
    "Ivory Coast": {
        "GK":  ["Yahia Fofana", "Mohamed Kone", "Alban Lafont"],
        "DEF": ["Emmanuel Agbadou", "Christopher Operi", "Ousmane Diomande", "Guela Doue",
                "Ghislain Konan", "Odilon Kossounou", "Wilfried Singo", "Evan Ndicka"],
        "MID": ["Seko Fofana", "Parfait Guiagon", "Christ Oulai", "Franck Kessie",
                "Ibrahim Sangare", "Jean Michael Seri"],
        "FWD": ["Simon Adingra", "Ange-Yoan Bonny", "Amad Diallo", "Oumar Diakite",
                "Yan Diomande", "Evann Guessand", "Nicolas Pepe", "Bazoumana Toure", "Elye Wahi"],
    },
    "Japan": {
        "GK":  ["Tomoki Hayakawa", "Keisuke Osako", "Zion Suzuki"],
        "DEF": ["Ko Itakura", "Hiroki Ito", "Yuto Nagatomo", "Ayumu Seko", "Yukinari Sugawara",
                "Junnosuke Suzuki", "Shogo Taniguchi", "Takehiro Tomiyasu", "Tsuyoshi Watanabe"],
        "MID": ["Ritsu Doan", "Wataru Endo", "Junya Ito", "Daichi Kamada", "Takefusa Kubo",
                "Keito Nakamura", "Kaishu Sano", "Ao Tanaka"],
        "FWD": ["Keisuke Goto", "Daizen Maeda", "Koki Ogawa", "Kento Shiogai",
                "Yuito Suzuki", "Ayase Ueda"],
    },
    "Jordan": {
        "GK":  ["Yazid Abulaila", "Noor Bani Attiah", "Abdallah Al Fakhouri"],
        "DEF": ["Mohammad Abu Hashish", "Abdullah Nasib", "Hussam Abu Dhahab", "Yazan Al Arab",
                "Mohammad Abu Alnadi", "Salem Obaid", "Saed Al Rosan", "Ehsan Haddad", "Anas Badawi"],
        "MID": ["Amer Jamous", "Noor Al Rawabdeh", "Rajaei Ayed", "Ibrahim Sadeh",
                "Mohannad Abu Taha", "Nizar Al Rashdan", "Mohammad Al Dawoud", "Mahmoud Mardahi"],
        "FWD": ["Mohammad Abu Zraiq", "Ali Olwan", "Mousa Al Tamari",
                "Odeh Fakhoury", "Ibrahim Sabra", "Ali Azaizeh"],
    },
    "Mexico": {
        "GK":  ["Raul Rangel", "Guillermo Ochoa", "Carlos Acevedo"],
        "DEF": ["Jorge Sanchez", "Israel Reyes", "Cesar Montes", "Johan Vasquez",
                "Jesus Gallardo", "Mateo Chavez", "Edson Alvarez"],
        "MID": ["Erik Lira", "Orbelin Pineda", "Alvaro Fidalgo", "Brian Gutierrez",
                "Luis Romo", "Obed Vargas", "Gilberto Mora", "Luis Chavez"],
        "FWD": ["Roberto Alvarado", "Cesar Huerta", "Alexis Vega", "Julian Quinones",
                "Guillermo Martinez", "Armando Gonzalez", "Santiago Gimenez", "Raul Jimenez"],
    },
    "Morocco": {
        "GK":  ["Yassine Bounou", "Munir El Kajoui", "Ahmed Reda Tagnaouti"],
        "DEF": ["Noussair Mazraoui", "Anas Salah-Eddine", "Youssef Bellammari", "Achraf Hakimi",
                "Zakaria El Ouahdi", "Nayef Aguerd", "Chadi Riad", "Redouane Halhal", "Issa Diop"],
        "MID": ["Samir El Mourabet", "Ayoub Bouaddi", "Neil El Aynaoui", "Sofyan Amrabat",
                "Azzedine Ounahi", "Bilal El Khannouss", "Ismael Saibari"],
        "FWD": ["Abdesamad Ezzalzouli", "Chemsdine Talbi", "Soufiane Rahimi",
                "Ayoub El Kaabi", "Brahim Diaz", "Yassine Gessim", "Ayoube Amaimouni"],
    },
    "Netherlands": {
        "GK":  ["Mark Flekken", "Robin Roefs", "Bart Verbruggen"],
        "DEF": ["Nathan Ake", "Virgil van Dijk", "Denzel Dumfries", "Jan Paul van Hecke",
                "Jurrien Timber", "Jorrel Hato", "Micky van de Ven"],
        "MID": ["Ryan Gravenberch", "Frenkie de Jong", "Teun Koopmeiners", "Tijjani Reijnders",
                "Marten de Roon", "Guus Til", "Quinten Timber", "Mats Wieffer"],
        "FWD": ["Brian Brobbey", "Memphis Depay", "Cody Gakpo", "Noa Lang",
                "Donyell Malen", "Crysencio Summerville", "Wout Weghorst", "Justin Kluivert"],
    },
    "New Zealand": {
        "GK":  ["Max Crocombe", "Alex Paulsen", "Michael Woud"],
        "DEF": ["Tyler Bindon", "Michael Boxall", "Liberato Cacace", "Francis de Vries",
                "Callan Elliot", "Tim Payne", "Nando Pijnaker", "Tommy Smith", "Finn Surman"],
        "MID": ["Lachlan Bayliss", "Joe Bell", "Matt Garbett", "Eli Just", "Callum McCowatt",
                "Ben Old", "Alex Rufer", "Marko Stamenic", "Sarpreet Singh", "Ryan Thomas"],
        "FWD": ["Kosta Barbarouses", "Chris Wood", "Hamish Watson", "Ben Waine", "Finn Surman"],
    },
    "Nigeria": {
        "GK":  ["Stanley Nwabali", "Maduka Okoye", "John Noble"],
        "DEF": ["Calvin Bassey", "Ola Aina", "William Troost-Ekong", "Chidozie Awaziem",
                "Zaidu Sanusi", "Semi Ajayi", "Kenneth Omeruo", "Bright Osayi-Samuel"],
        "MID": ["Frank Onyeka", "Alex Iwobi", "Samuel Chukwueze", "Wilfred Ndidi",
                "Raphael Onyedika", "Alhassan Yusuf", "Joe Aribo"],
        "FWD": ["Victor Osimhen", "Taiwo Awoniyi", "Kelechi Iheanacho",
                "Ademola Lookman", "Terem Moffi", "Paul Onuachu"],
    },
    "Norway": {
        "GK":  ["Orjan Nyland", "Ørjan Hansen", "David Haikin"],
        "DEF": ["Leo Ostigard", "Stian Gregersen", "Birger Meling", "Julian Ryerson",
                "Andreas Hanche-Olsen", "Bard Finne", "Kristian Thorstvedt"],
        "MID": ["Martin Odegaard", "Sander Berge", "Mathias Normann", "Fredrik Aursnes",
                "Patrick Berg", "Mats Moller Daehli"],
        "FWD": ["Erling Haaland", "Alexander Sorloth", "Jorgen Strand Larsen",
                "Antonio Nusa", "Mohamed Elyounoussi"],
    },
    "Panama": {
        "GK":  ["Orlando Mosquera", "Luis Mejia", "Gianluca Romero"],
        "DEF": ["Fidel Escobar", "Harold Cummings", "Eric Davis", "Roderick Miller",
                "Michael Murillo", "Cesar Blackman", "Andres Andrade"],
        "MID": ["Adalberto Carrasquilla", "Anibal Godoy", "Cristian Martinez",
                "Edgardo Farina", "Jesus Poveda", "Jose Fajardo", "Freddy Gondola"],
        "FWD": ["Ismael Diaz", "Cecilio Waterman", "Jose Rodriguez",
                "Gabriel Torres", "Maximiliano Urruti", "Alfredo Stephens"],
    },
    "Paraguay": {
        "GK":  ["Antony Silva", "Alfredo Aguilar", "Rodrigo Muñoz"],
        "DEF": ["Junior Alonso", "Gustavo Gomez", "Omar Alderete", "Fabian Balbuena",
                "Jorge Morel", "Santiago Arzamendia", "Ivan Ramirez"],
        "MID": ["Miguel Almiron", "Andres Cubas", "Gabriel Avalos", "Mathias Villasanti",
                "Richard Ortiz", "Braian Ojeda", "Damian Bobadilla"],
        "FWD": ["Julio Enciso", "Antonio Sanabria", "Alejandro Romero Gamarra",
                "Jesus Medina", "Robert Morales"],
    },
    "Portugal": {
        "GK":  ["Diogo Costa", "Rui Silva", "Jose Sa"],
        "DEF": ["Ruben Dias", "Joao Cancelo", "Nelson Semedo", "Nuno Mendes",
                "Diogo Dalot", "Goncalo Inacio", "Renato Veiga", "Tomas Araujo"],
        "MID": ["Joao Palhinha", "Ruben Neves", "Bruno Fernandes", "Vitinha",
                "Joao Neves", "Bernardo Silva"],
        "FWD": ["Cristiano Ronaldo", "Rafael Leao", "Pedro Neto",
                "Diogo Jota", "Goncalo Ramos", "Francisco Conceicao"],
    },
    "Qatar": {
        "GK":  ["Meshaal Barsham", "Saad Al Sheeb", "Yousuf Hassan"],
        "DEF": ["Boualem Khoukhi", "Pedro Miguel", "Sultan Al Brake", "Tarek Salman",
                "Al-Hashmi Al-Hussain", "Ayoub Al-Alawi", "Bassam Al-Rawi",
                "Rayyan Al-Ali", "Issa Laye", "Lucas Mendes", "Mohammed Waad", "Niall Mason"],
        "MID": ["Ahmed Fathi", "Jassim Gaber", "Assim Madibo", "Abdulaziz Hatem",
                "Karim Boudiaf", "Mohammed Mannai", "Homam Al-Amin"],
        "FWD": ["Almoez Ali", "Akram Afif", "Tahsin Mohammed", "Edmilson Junior",
                "Ahmed Al-Ganehi", "Ahmed Alaa", "Sebastian Soria",
                "Hassan Al-Haydos", "Mohammed Muntari"],
    },
    "Saudi Arabia": {
        "GK":  ["Mohammed Al-Owais", "Nawaf Al-Aqidi", "Mohammed Al-Yami"],
        "DEF": ["Ali Al-Bulayhi", "Sultan Al-Ghannam", "Abdulelah Al-Amri", "Hassan Al-Tambakti",
                "Mohammed Al-Breik", "Saud Abdulhamid", "Waleed Al-Shahrani"],
        "MID": ["Mohamed Kanno", "Salman Al-Faraj", "Riyadh Sharahili", "Abdullah Al-Hamdan",
                "Mohammed Al-Qasem", "Nasser Al-Dawsari", "Ali Al-Hassan"],
        "FWD": ["Firas Al-Buraikan", "Salem Al-Dawsari", "Haitham Asiri",
                "Saleh Al-Shehri", "Ayman Yahya", "Abdullah Radif"],
    },
    "Scotland": {
        "GK":  ["Angus Gunn", "Craig Gordon", "Liam Kelly"],
        "DEF": ["Andrew Robertson", "Kieran Tierney", "Scott McKenna", "Grant Hanley",
                "Jack Hendry", "Anthony Ralston", "Aaron Hickey"],
        "MID": ["Callum McGregor", "John McGinn", "Scott McTominay", "Billy Gilmour",
                "Ryan Christie", "Stuart Armstrong", "Kenny McLean"],
        "FWD": ["Lyndon Dykes", "Lawrence Shankland", "Ryan Gauld",
                "Che Adams", "Liel Abada", "Jacob Brown"],
    },
    "Senegal": {
        "GK":  ["Edouard Mendy", "Seny Dieng", "Alfred Gomis"],
        "DEF": ["Kalidou Koulibaly", "Abdou Diallo", "Youssouf Sabaly",
                "Formose Mendy", "Moussa Niakhate"],
        "MID": ["Idrissa Gueye", "Nampalys Mendy", "Pape Matar Sarr",
                "Lamine Camara", "Pathe Ciss"],
        "FWD": ["Sadio Mane", "Ismaila Sarr", "Nicolas Jackson",
                "Habib Diallo", "Boulaye Dia"],
    },
    "South Africa": {
        "GK":  ["Ronwen Williams", "Veli Mothwa", "Ricardo Goss"],
        "DEF": ["Siyanda Xulu", "Rushine de Reuck", "Thibang Phete", "Terrence Mashego",
                "Nkosinathi Sibisi", "Mothobi Mvala", "Grant Kekana"],
        "MID": ["Themba Zwane", "Teboho Mokoena", "Ethan Nwaneri", "Iqraam Rayners",
                "Yusuf Maart", "Lyle Foster", "Evidence Makgopa"],
        "FWD": ["Percy Tau", "Lebo Mothiba", "Bradley Grobler",
                "Bafana Khuluse", "Khulekani Kubheka"],
    },
    "South Korea": {
        "GK":  ["Kim Seung-gyu", "Cho Hyun-woo", "Song Bum-keun"],
        "DEF": ["Kim Min-jae", "Lee Ki-je", "Kim Jin-su", "Seol Young-woo",
                "Hong Chul", "Hwang Hyun-beom", "Kim Tae-hwan"],
        "MID": ["Son Heung-min", "Lee Jae-sung", "Jung Woo-young", "Hwang In-beom",
                "Lee Kang-in", "Paik Seung-ho", "Cho Gue-sung"],
        "FWD": ["Oh Hyeon-gyu", "Hwang Hee-chan", "Um Won-sang", "Bae Jun-ho", "Lee Seung-won"],
    },
    "Sweden": {
        "GK":  ["Robin Olsen", "Karl-Johan Johnsson", "Samuel Brolin"],
        "DEF": ["Victor Lindelof", "Isak Hien", "Ludwig Augustinsson", "Emil Krafth",
                "Joakim Nilsson", "Filip Helander", "Oliver Dovin"],
        "MID": ["Albin Ekdal", "Dejan Kulusevski", "Viktor Gyokeres", "Gustav Isaksen",
                "Mattias Svanberg", "Samuel Dahl", "Kristoffer Olsson"],
        "FWD": ["Alexander Isak", "Jordan Larsson", "Anthony Elanga",
                "Viktor Gyokeres", "Marcus Danielson"],
    },
    "Switzerland": {
        "GK":  ["Gregor Kobel", "Yvon Mvogo", "Marvin Keller"],
        "DEF": ["Manuel Akanji", "Fabian Schar", "Ricardo Rodriguez", "Silvan Widmer",
                "Nico Elvedi", "Kevin Mbabu", "Eray Comert"],
        "MID": ["Granit Xhaka", "Remo Freuler", "Denis Zakaria", "Ruben Vargas",
                "Xherdan Shaqiri", "Michel Aebischer", "Edimilson Fernandes"],
        "FWD": ["Breel Embolo", "Haris Seferovic", "Noah Okafor",
                "Zeki Amdouni", "Dan Ndoye", "Fabian Rieder"],
    },
    "Tunisia": {
        "GK":  ["Aymen Dahmen", "Bechir Ben Said", "Moez Ben Cherifia"],
        "DEF": ["Montassar Talbi", "Dylan Bronn", "Wajdi Kechrida", "Ali Maaloul",
                "Nader Ghandri", "Bilel Ifa", "Mohamed Drager"],
        "MID": ["Hannibal Mejbri", "Anis Ben Slimane", "Ellyes Skhiri", "Wahbi Khazri",
                "Naim Sliti", "Ghaylen Chaalali", "Ferjani Sassi"],
        "FWD": ["Youssef Msakni", "Seifeddine Jaziri", "Saad Bguir",
                "Taha Yassine Khenissi", "Mohamed Ali Ben Romdhane"],
    },
    "Turkey": {
        "GK":  ["Mert Gunok", "Ugurcan Cakir", "Altay Bayindir"],
        "DEF": ["Samet Akaydin", "Merih Demiral", "Zeki Celik", "Ferdi Kadioglu",
                "Abdulkerim Bardakci", "Ridvan Yilmaz", "Ahmetcan Kaplan"],
        "MID": ["Hakan Calhanoglu", "Salih Ozcan", "Kaan Ayhan", "Okay Yokuslu",
                "Orkun Kokcu", "Ismail Yuksek", "Yunus Akgun"],
        "FWD": ["Arda Guler", "Kerem Akturkoglu", "Yusuf Yazici",
                "Cenk Tosun", "Baris Alper Yilmaz", "Serdar Dursun"],
    },
    "Uruguay": {
        "GK":  ["Sergio Rochet", "Sebastian Sosa", "Guillermo De Amores"],
        "DEF": ["Jose Maria Gimenez", "Diego Godin", "Mathias Olivera", "Nahitan Nandez",
                "Sebastian Caceres", "Ronald Araujo", "Agustin Rogel"],
        "MID": ["Federico Valverde", "Manuel Ugarte", "Rodrigo Bentancur", "Lucas Torreira",
                "Nicolas De La Cruz", "Facundo Pellistri", "Matias Vecino"],
        "FWD": ["Darwin Nunez", "Luis Suarez", "Edinson Cavani", "Maximiliano Gomez",
                "Brian Rodriguez", "Agustin Alvarez Martinez"],
    },
    "USA": {
        "GK":  ["Matt Turner", "Ethan Horvath", "Patrick Schulte"],
        "DEF": ["Sergino Dest", "Antonee Robinson", "Tim Ream", "Chris Richards",
                "Walker Zimmerman", "Joe Scally", "Cameron Carter-Vickers"],
        "MID": ["Tyler Adams", "Weston McKennie", "Yunus Musah", "Gio Reyna",
                "Brenden Aaronson", "Malik Tillman"],
        "FWD": ["Christian Pulisic", "Josh Sargent", "Ricardo Pepi",
                "Tim Weah", "Folarin Balogun", "Cade Cowell"],
    },
    "Uzbekistan": {
        "GK":  ["Eldorbek Smatov", "Oybek Khudoyberdiev", "Jasurbek Yakhshiboev"],
        "DEF": ["Dostonbek Khamdamov", "Sherzod Nasrullayev", "Sanjar Tursunov",
                "Abdukodir Khusanov", "Temur Kapadze", "Bobur Abdixoliqov"],
        "MID": ["Jamshid Iskanderov", "Otabek Shukurov", "Khojiakbar Alijonov",
                "Jaloliddin Masharipov", "Dilshod Hamrobekov", "Khumoyun Qodirov"],
        "FWD": ["Eldor Shomurodov", "Dostonbek Tursunov", "Ilhomjon Hamrobekov",
                "Akbar Djuraev", "Umid Nishonov", "Bekhruz Mirzaev"],
    },
}

TOURNAMENT = "FIFA World Cup 2026"
TOURNAMENT_DATE = "2026-06-11"


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'\s*\(\d+\)\s*$', '', name)
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ä': 'a', 'ö': 'o', 'ü': 'u', 'ñ': 'n', 'ç': 'c',
        'ž': 'z', 'š': 's', 'ć': 'c', 'č': 'c', 'ø': 'o',
        'å': 'a', 'æ': 'ae', 'ı': 'i', 'ğ': 'g', 'ş': 's',
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    return name


def build_squads_csv(profiles_path: str, output_path: str, min_score: int = 75):
    print(f"Cargando perfiles: {profiles_path}")
    profiles = pd.read_csv(
        profiles_path,
        usecols=['player_id', 'player_name', 'citizenship', 'main_position', 'date_of_birth']
    )

    profiles['player_name'] = profiles['player_name'].fillna('')

    profiles['name_norm'] = profiles['player_name'].apply(normalize)
    profiles['citizenship_norm'] = profiles['citizenship'].str.lower().str.strip().fillna('')

    name_list = profiles['name_norm'].tolist()

    records = []
    unmatched = []

    for team, positions in SQUADS_RAW.items():
        print(f"\n── {team} ──")
        country_variants = [team.lower()]
        mask = profiles['citizenship_norm'].isin(country_variants)
        candidates = profiles[mask] if mask.sum() >= 15 else profiles
        cand_names = candidates['name_norm'].tolist()
        cand_index = candidates.index.tolist()

        for pos, players in positions.items():
            for player_name in players:
                name_norm = normalize(player_name)

                result = process.extractOne(name_norm, cand_names, scorer=fuzz.token_sort_ratio) if cand_names else None

                if result and result[1] >= min_score:
                    matched_idx = cand_index[cand_names.index(result[0])]
                    row = profiles.loc[matched_idx]
                    records.append({
                        'team_name': team, 'player_name': player_name, 'position': pos,
                        'player_id': int(row['player_id']), 'tm_name': row['player_name'],
                        'citizenship': row['citizenship'], 'date_of_birth': row['date_of_birth'],
                        'match_score': result[1], 'tournament': TOURNAMENT, 'tournament_date': TOURNAMENT_DATE,
                    })
                    print(f"  [{pos}] ✓ {player_name} → {row['player_name']} ({result[1]})")
                else:
                    result_g = process.extractOne(name_norm, name_list, scorer=fuzz.token_sort_ratio)
                    if result_g and result_g[1] >= min_score:
                        matched_idx = name_list.index(result_g[0])
                        row = profiles.iloc[matched_idx]
                        records.append({
                            'team_name': team, 'player_name': player_name, 'position': pos,
                            'player_id': int(row['player_id']), 'tm_name': row['player_name'],
                            'citizenship': row['citizenship'], 'date_of_birth': row['date_of_birth'],
                            'match_score': result_g[1], 'tournament': TOURNAMENT, 'tournament_date': TOURNAMENT_DATE,
                        })
                        print(f"  [{pos}] ~ {player_name} → {row['player_name']} [global, {result_g[1]}]")
                    else:
                        unmatched.append({'team': team, 'player': player_name, 'pos': pos})
                        print(f"  [{pos}] ✗ {player_name} — SIN MATCH")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_path, index=False)
    print(f"\n✅ CSV guardado: {output_path} ({len(df_out)} jugadores)")

    if unmatched:
        unmatched_path = output_path.replace('.csv', '_unmatched.csv')
        pd.DataFrame(unmatched).to_csv(unmatched_path, index=False)
        print(f"⚠️  {len(unmatched)} sin match → {unmatched_path}")

    print(f"\n── Resumen ──")
    print(f"  Equipos:         {len(SQUADS_RAW)}/48")
    print(f"  Matcheados:      {len(records)}")
    print(f"  Sin match:       {len(unmatched)}")
    if records:
        scores = [r['match_score'] for r in records]
        print(f"  Score medio:     {sum(scores)/len(scores):.1f}")
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', default='data/football-datasets/datalake/transfermarkt/player_profiles/player_profiles.csv')
    parser.add_argument('--output', default='data/squads/convocatorias_oficiales.csv')
    parser.add_argument('--min-score', type=int, default=75)
    args = parser.parse_args()
    build_squads_csv(args.profiles, args.output, args.min_score)