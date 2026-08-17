import typer

from cdcn_msu_mfsd import __version__

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo(__version__)


if __name__ == "__main__":
    app()

