import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Transaction:
    """Clase para almacenar una transacción financiera."""
    date: str
    transaction: str
    document: str
    place: str
    debits: str
    credits: str
    balance: str

class TransactionCleaner:
    """Clase para limpiar y procesar transacciones financieras."""
    
    MONTHS = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']
    MONTH_PATTERN = re.compile(r'^(' + '|'.join(MONTHS) + r') \d{1,2}')
    AMOUNT_PATTERN = re.compile(r'\d{1,3}(,\d{3})*\.\d{2}')
    
    KNOWN_PLACES = {
        'REDEBAN', 'DOMICILIACIONES', 'ACH', 'INTERNET', 'ASCREDIBANCO', 'CALLE 53', 
        'RED PROPIA', 'PRADO VERANIEGO', 'ZIPAQUIRA', 'SERVIBANCA', 'ZIPAQUIRA III', 
        'QUIRIGUA III', 'CC FONTANAR', 'CENTRO CHIA', 'AVENIDA EL DORADO', 
        'EXITO QUIRIGUA', 'TEXACO 26', 'CLINICA COUNTRY'
    }
    
    NO_DOCUMENT_TRANSACTIONS = {
        'DEBITO POR RECAUDO',
        'ABONO DE INTERESES',
        'GRAVAMEN MOVS FINANCIEROS',
        'IVA SOBRE COMISIONES',
        'ABONO NOMINA'
    }

    @staticmethod
    def clean_text(text: str) -> str:
        """Limpia y normaliza el texto."""
        return ' '.join(text.split())

    def find_place(self, text: str) -> str:
        """Encuentra el lugar de la transacción."""
        best_match = None
        best_index = -1
        
        for place in self.KNOWN_PLACES:
            pattern = r'\b' + re.escape(place) + r'\b'
            match = re.search(pattern, text)
            if match and match.start() > best_index:
                best_index = match.start()
                best_match = place
        
        return best_match or ''

    def parse_amounts(self, text: str) -> tuple:
        """Extrae los montos de la transacción."""
        amounts = re.findall(self.AMOUNT_PATTERN, text)
        
        if not amounts:
            return '', '', ''
        
        if len(amounts) == 1:
            return '', '', amounts[0]
            
        debit = ''
        credit = ''
        if re.search(r'ABONO|CREDITO|INTERESES', text, re.IGNORECASE):
            credit = amounts[0]
        else:
            debit = amounts[0]
            
        return debit, credit, amounts[-1]

    def parse_transaction(self, line: str) -> Optional[Transaction]:
        """Procesa una línea de transacción y retorna un objeto Transaction."""
        match = self.MONTH_PATTERN.match(line)
        if not match:
            return None
            
        date = match.group()
        body = line[len(date):].strip()
        
        debit, credit, balance = self.parse_amounts(body)
        body = self.AMOUNT_PATTERN.sub('', body).strip()
        body = self.clean_text(body)
        
        place = self.find_place(body)
        if place:
            body = re.sub(r'\b' + re.escape(place) + r'\b', '', body).strip()
            body = self.clean_text(body)
            
        document = ''
        transaction = body
        
        # Determinar si la transacción debe tener número de documento
        has_document = not any(body.startswith(prefix) for prefix in self.NO_DOCUMENT_TRANSACTIONS)
        
        if has_document and body:
            words = body.split()
            if words:
                document = words[-1]
                transaction = ' '.join(words[:-1])
        
        return Transaction(date, transaction, document, place, debit, credit, balance)

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Procesa el archivo de entrada y genera el archivo limpio."""
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Unir líneas que pertenecen a la misma transacción
        lines = content.splitlines()
        transactions = []
        current_transaction = None
        
        for line in lines:
            line = line.strip()
            if self.MONTH_PATTERN.match(line):
                if current_transaction is not None:
                    transactions.append(current_transaction)
                current_transaction = line
            elif current_transaction is not None:
                current_transaction += ' ' + line
        
        if current_transaction is not None:
            transactions.append(current_transaction)
        
        # Procesar transacciones y escribir resultado
        output_lines = ["Fecha,Transacción,Documento,Lugar,Débitos,Créditos,Saldos"]
        
        for trans in transactions:
            if ',' in trans:  # Si ya está en formato CSV
                fields = [f.strip() for f in trans.split(',')]
                fields = (fields + [''] * 7)[:7]  # Asegurar 7 campos
                output_lines.append(','.join(fields))
            else:
                parsed = self.parse_transaction(trans)
                if parsed:
                    output_lines.append(f"{parsed.date},{parsed.transaction},{parsed.document},"
                                     f"{parsed.place},{parsed.debits},{parsed.credits},{parsed.balance}")
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write('\n'.join(output_lines))

def main():
    """Función principal."""
    cleaner = TransactionCleaner()
    input_path = Path('data/raw/Datos Movimientos Financieros.txt')
    output_path = Path('data/processed/Datos Movimientos Financieros Ajustados.csv')
    
    # Crear directorio si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cleaner.process_file(input_path, output_path)

if __name__ == '__main__':
    main()
