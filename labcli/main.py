import typer
from labcli.commands import train, play

app = typer.Typer(
    name='labcli',
    help='Lab CLI tool',
    no_args_is_help=True,
)

app.add_typer(train.app)
app.add_typer(play.app)
