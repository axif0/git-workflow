from rich.console import Console
from rich.table import Table

console = Console()

class ComplementEngine:
    def __init__(self, value: str, base: int):
        # auto-detect if input is binary, octal, hex, or decimal
        if value.startswith("0b"):
            self.number = int(value, 2)
        elif value.startswith("0o"):
            self.number = int(value, 8)
        elif value.startswith("0x"):
            self.number = int(value, 16)
        else:
            self.number = int(value)  # decimal
        
        self.base = base
        self.digits = len(str(self.number))

    def r_minus_1(self):
        return int(''.join(str(self.base - 1 - int(d)) for d in str(self.number)))
    
    def r_complement(self):
        return self.base ** self.digits - self.number

    def explain(self):
        table = Table(title=f"[bold cyan]Complement Calculation (Base {self.base})[/bold cyan]")
        table.add_column("Step", style="bold green")
        table.add_column("Details", style="white")

        table.add_row("Number", str(self.number))
        table.add_row("Digits", str(self.digits))
        table.add_row("r^n", f"{self.base}^{self.digits} = {self.base ** self.digits}")
        table.add_row("(r-1)'s complement", str(self.r_minus_1()))
        table.add_row("r's complement", str(self.r_complement()))

        console.print(table)


# -----------------------------
# Interactive Menu
# -----------------------------
if __name__ == "__main__":
    console.print("[bold magenta]🔥 Welcome to Sexy Complement Engine 🔥[/bold magenta]\n")
    num = input("Enter a number (supports 15, 0b1111, 0o17, 0xF): ")
    base = int(input("Enter the base (e.g., 2, 8, 10, 13, 16): "))

    engine = ComplementEngine(num, base)
    engine.explain()
