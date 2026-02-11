from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent 
DATA_FOLDER = BASE_DIR / "data"

MAX_GROUP_CHARS = 24_000

DISCLAIMER = "\n\n\n**Le informazioni sono state estratte dal testo fornito e potrebbero essere incomplete.**"

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key" 

STANDARD_PROMPT = (
    'Sei un assistente legale esperto. Riceverai appunti estratti da vari documenti relativi alla stessa pratica (es. contratto originale + atti successivi).\n'

    'OBIETTIVO:\n'
    'Compilare un report strutturato che rappresenti la SITUAZIONE ATTUALE E AGGIORNATA.\n'
    'Devi incrociare le informazioni: se un documento successivo (es. atto di decesso, appendice) modifica quello precedente, l\'informazione valida è l\'ultima cronologicamente dal punto di vista temporale.\n'
    # 'Leggere l’intero documento ed estrarre SEMPRE le informazioni richieste, indicando SEMPRE per ciascuna:\n'
    # '- la pagina e il nome del file. Se le informazioni sono sparse in più pagine, indicale tutte.\n'
    # '- l\'infomazione ricavata\n'
    # '- una citazione testuale\n'
    # '- se un dato che analizzi tra quelli richiesto ti sembra abbia un formato strano, indicalo comunque aggiungendo il flag [POSSIBILE SCRITTA A MANO]\n'

    # 'REGOLE GENERALI:\n'
    # '- Non inventare nulla.\n'
    # '- Usa formulazioni il più possibile fedeli al testo originale.\n'
    # '- Se un’informazione non è esplicitamente presente, rispondi “Non presente”.\n'
    # '- Se sono presenti più valori (durate, parti, terreni, pagamenti), elencali TUTTI.\n'
    # '- Restituisci SEMPRE l’output in formato tabellare, una tabella per sezione.\n'

    'SEZIONI DA ESTRARRE:\n'

    '1) DATI GENERALI DEL CONTRATTO\n'
    '- Numero di Repertorio/Trascrizione\n'
    '- Numero di Raccolta\n'
    '- Data di sottoscrizione (formato GG/MM/AAAA)\n'
    '- Breve citazione\n'
    '- Pagina nella quale si trova l\'informazione\n'
    '- File nella quale si trova l\'informazione\n'
    '- flag [POSSIBILE SCRITTA A MANO] se il dato riportato ha un formato strano\n'

    '2) DURATE E SCADENZE CONTRATTUALI:\n'
    'Individua TUTTE le durate e scadenze presenti (preliminare, definitivo, diritti reali, opzioni, proroghe, rinnovi, accordi accessori).\n'
    'Per ciascuna indica:\n'
    '- Tipo di rapporto\n'
    '- Durata\n'
    '- Dies a quo\n'
    '- Condizioni o estensioni\n'
    '- Breve citazione\n'
    '- Pagina nella quale si trova l\'informazione\n'
    '- File nella quale si trova l\'informazione\n'

    '3) DURATA DEL RINNOVO:\n'
    'Durata di eventuali rinnovi o proroghe (automatiche o facoltative), se previste.\n'
    '- Breve citazione\n'
    '- Pagina nella quale si trova l\'informazione\n'
    '- File nella quale si trova l\'informazione\n'

    '4) OGGETTO DEL CONTRATTO:\n'
    'Assegna una o più categorie, se presenti:\n'
    'Locazione, Diritti di superficie, Diritti di servitù, Esproprio, Occupazione temporanea, Compravendita, Royalty, oppure “Non specificato”.\n'
    '- Breve citazione\n'
    '- Pagina nella quale si trova l\'informazione\n'
    '- File nella quale si trova l\'informazione\n'

    '5) BENEFICIARI (PARTI CONTRAENTI):\n'
    'Per ciascuna persona fisica o giuridica:\n'
    '- Nome / Ragione sociale\n'
    '- Data e luogo di nascita (se persona fisica)\n'
    '- Indirizzo\n'
    '- Codice Fiscale\n'
    '- IBAN\n'
    '- Numero di telefono\n'
    '- Pagina nella quale si trova l\'informazione\n'
    '- File nella quale si trova l\'informazione\n'

    '6) INFORMAZIONI SUI TERRENI:\n'
    'Per ogni foglio/particella:\n'
    '- Foglio e particella\n'
    '- Comune\n'
    '- Estensione\n'
    '- Categoria e classe catastale\n'
    '- R.D. e R.A.\n'
    '- Tipo di proprietà\n'
    '- Quota di proprietà\n'
    '- Pagina nella quale si trova l\'informazione\n'
    '- File nella quale si trova l\'informazione\n'

    '7) INFORMAZIONI SUL PAGAMENTO:\n'
    'Devi elencare TUTTI i movimenti economici presenti nelle note, distinguendoli chiaramente.\n'
    'NON SOVRASCRIVERE il "Prezzo di Vendita" con le "Imposte" o "Tasse" anche se compaiono in pagine successive.\n'
    'Crea righe separate per:\n'
    '- Oggetto del pagamento\n'
    '- Corrispettivo\n'
    '- Tassa di registrazione (%)\n'
    '- Beneficiario\n'
    '- Eventuali termini di ritardo\n'
    '- Breve citazione\n'
    '- Pagina nella quale si trova l\'informazione\n'
    '- File nella quale si trova l\'informazione\n'

    '8) EVENTUALI ALTRE INFORMAZIONI RILEVANTI\n'
    '- Qui se per i beneficiari non trovi un IBAN, fornisci se presente, tutti gli IBAN presenti nel documento. Non è detto che nella stessa sezione del documento in cui '
    'trovi nel informazioni dei beneficiari trovi anche l\'informazione dell\'IBAN\n'
    '- Pagina nella quale si trova l\'informazione\n'
    '- File nella quale si trova l\'informazione\n'

    # 'FORMATO DI USCITA:\n'
    # '- Tabelle separate per ciascuna sezione\n'
    # '- Nessuna informazione diversa da quelle esplicitamente richieste\n'
)

MAP_PROMPT_TEXT = (
    "Sei un analista legale. Analizza il seguente segmento di testo estratto da un contratto.\n"
    "Il tuo compito è ESTRARRE GREZZAMENTE qualsiasi informazione relativa ai seguenti punti:\n"

    "1. DATI CONTRATTUALI STANDARD (se presenti):\n"
    "- Dati del contratto (Numero di Trascrizione/Repertorio, Raccolta, date)\n"
    "- Durate, scadenze, rinnovi (tipo di rapporto, durata, dies a quo, condizioni o estensioni)\n"
    "- Oggetto del contratto (locazione, diritti di superficie, diritti di servitù, esproprio, occupazione temporanea, compravendita)\n"
    "- Anagrafiche parti (nomi, ragione sociale, data e luogo di nascita, codice fiscale, iban, numero di telefono, indirizzi)\n"
    "- Dati catastali terreni/immobili per ogni coppia foglio particella (comune, estensione, categoria e classe catastale, R.D. e R.A., tipo di proprietà, quota di proprietà)\n"
    "- Tutti i corrispettivi e pagamenti, anche eventuali pagamenti ricorrenti da fare per X anni (oggetto del pagamento, corrispettivo, tassa di registrazione beneficiario, eventuali termini di ritardo)\n\n"
    # "- è possibile che il contenuto del testo non sia completamente rilevante alle informazioni di un contratto legale. Fai una sintesi del suo contenuto seguendo comunque le istruzioni qui di seguito."
    
    "2. EVENTI MODIFICATIVI O INTEGRATIVI (Fondamentale):\n"
    "Cerca esplicitamente informazioni su:\n"
    "- DECESSI O SUCCESSIONI: Chi è deceduto? Chi sono gli eredi? Da che data?\n"
    "- VARIAZIONI DI PAGAMENTO: Nuovi IBAN, nuovi beneficiari, nuove modalità.\n"
    "- SUBENTRI O CESSIONI: Cambi nella titolarità del contratto.\n"
    "- MODIFICHE AI PATTI: Variazioni di canone, proroghe aggiuntive.\n\n"

    "ISTRUZIONI:\n"
    "- Se trovi un dato, trascrivilo citando SEMPRE il [FILE: ...] e la [PAGINA ...].\n"
    "- Se trovi un atto di decesso o una variazione, descrivi chiaramente: 'Il file X indica che in data Y è successo Z'.\n"
    "- Non preoccuparti di collegare i fatti ora, estrai solo le informazioni grezze.\n"
    "- Non aggiungere nient'altro che non sia esplicitamente chiesto\n\n"
    "TESTO:\n{context}"
)