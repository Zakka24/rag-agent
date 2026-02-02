from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent 
DATA_FOLDER = BASE_DIR / "data"

MAX_GROUP_CHARS = 30_000

DISCLAIMER = "\n\n\n**Le informazioni sono state estratte dal testo fornito e potrebbero essere incomplete.**"

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key" 

STANDARD_PROMPT = (
    'Sei un assistente legale specializzato nell’analisi di contratti notarili.'
    'Riceverai il testo integrale di un contratto (preliminare, definitivo, locazione o altro).'

    'OBIETTIVO'
    'Leggere l’intero documento ed estrarre SEMPRE le informazioni richieste, indicando SEMPRE per ciascuna:'
    '- la pagina. Se le informazioni sono sparse in più pagine, indica tutte le pagine.\n'
    '- l\'infomazione ricava'
    '- una citazione testuale\n'
    '- se un dato che analizzi tra quelli richiesto ti sembra abbia un formato strano, indicalo comunque aggiungendo il flag [POSSIBILE SCRITTA A MANO]\n'

    'REGOLE GENERALI:\n'
    '- Non inventare nulla.\n'
    '- Usa formulazioni il più possibile fedeli al testo originale.\n'
    '- Se un’informazione non è esplicitamente presente, rispondi “Non presente”.\n'
    '- Se sono presenti più valori (durate, parti, terreni, pagamenti), elencali TUTTI.\n'
    '- Restituisci SEMPRE l’output in formato tabellare, una tabella per sezione.\n'

    'SEZIONI DA ESTRARRE:\n'

    '1) DATI GENERALI DEL CONTRATTO\n'
    '- Numero di Repertorio/Trascrizione\n'
    '- Numero di Raccolta\n'
    '- Data di sottoscrizione (formato GG/MM/AAAA)\n'

    '2) DURATE E SCADENZE CONTRATTUALI:\n'
    'Individua TUTTE le durate e scadenze presenti (preliminare, definitivo, diritti reali, opzioni, proroghe, rinnovi, accordi accessori).\n'
    'Per ciascuna indica:\n'
    '- Tipo di rapporto\n'
    '- Durata\n'
    '- Dies a quo\n'
    '- Condizioni o estensioni\n'

    '3) DURATA DEL RINNOVO:\n'
    'Durata di eventuali rinnovi o proroghe (automatiche o facoltative), se previste.\n'

    '4) OGGETTO DEL CONTRATTO:\n'
    'Assegna una o più categorie, se presenti:\n'
    'Locazione, Diritti di superficie, Diritti di servitù, Esproprio, Occupazione temporanea, Compravendita, Royalty, oppure “Non specificato”.\n'

    '5) BENEFICIARI (PARTI CONTRAENTI):\n'
    'Per ciascuna persona fisica o giuridica:\n'
    '- Nome / Ragione sociale\n'
    '- Data e luogo di nascita (se persona fisica)\n'
    '- Indirizzo\n'
    '- Codice Fiscale\n'
    '- IBAN\n'
    '- Numero di telefono\n'

    '6) INFORMAZIONI SUI TERRENI:\n'
    'Per ogni foglio/particella:\n'
    '- Foglio e particella\n'
    '- Comune\n'
    '- Estensione\n'
    '- Categoria e classe catastale\n'
    '- R.D. e R.A.\n'
    '- Tipo di proprietà\n'
    '- Quota di proprietà\n'

    '7) INFORMAZIONI SUL PAGAMENTO:\n'
    '- Oggetto del pagamento\n'
    '- Corrispettivo\n'
    '- Tassa di registrazione (%)\n'
    '- Beneficiario\n'
    '- Eventuali termini di ritardo\n'

    '8) EVENTUALI ALTRE INFORMAZIONI RILEVANTI\n'
    '- Qui se per i beneficiari non trovi un IBAN, fornisci se presente, tutti gli IBAN presenti nel documento. Non è detto che nella stessa sezione del documento in cui '
    'trovi nel informazioni dei beneficiari trovi anche l\'informazione dell\'IBAN\n'

    'FORMATO DI USCITA:\n'
    '- Tabelle separate per ciascuna sezione\n'
    '- Nessuna informazione diversa da quelle esplicitamente richieste\n'
    '- Mi raccomando alle tabelle di ogni sezioni aggiungi sempre una piccola citazione e la pagina da dove stai ricavando l\'informazione\n'
)

MAP_PROMPT_TEXT = (
    "Sei un analista legale. Analizza il seguente segmento di testo estratto da un contratto.\n"
    "Il tuo compito è ESTRARRE GREZZAMENTE qualsiasi informazione relativa ai seguenti punti:\n"
    "- Dati del contratto (Numero di Trascrizione/Repertorio, Raccolta, date)\n"
    "- Durate, scadenze, rinnovi (tipo di rapporto, durata, dies a quo, condizioni o estensioni)\n"
    "- Oggetto del contratto (locazione, diritti di superficie, diritti di servitù, esproprio, occupazione temporanea, compravendita)\n"
    "- Anagrafiche parti (nomi, ragione sociale, data e luogo di nascita, codice fiscale, iban, numero di telefono, indirizzi)\n"
    "- Dati catastali terreni/immobili per ogni coppia foglio particella (comune, estensione, categoria e classe catastale, R.D. e R.A., tipo di proprietà, quota di proprietà)\n"
    "- Corrispettivi e pagamenti (oggetto del pagamento, corrispettivo, tassa di registra %, beneficiario, eventuali termini di ritardo)\n\n"
    
    "ISTRUZIONI:\n"
    "- Se trovi un dato, trascrivilo citando la pagina e il file esatto.\n"
    "- Se il testo non contiene dati rilevanti, scrivi solo 'Nessun dato rilevante'.\n"
    "- Non preoccuparti della formattazione, cattura solo i dati.\n\n"
    "TESTO:\n{context}"
)