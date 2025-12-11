import click
from PIL import Image
from mylib.logic import predict_image_class, resize_image

@click.group()
def cli():
    pass

@cli.command()
@click.argument('filepath')
def predict(filepath):
    """Predice la clase de un archivo de imagen local."""
    try:
        # Abrimos la imagen usando PIL (como sugiere el ejemplo pag 6)
        image = Image.open(filepath)
        result = predict_image_class(image)
        click.echo(f"La imagen es un: {result}")
    except Exception as e: # pylint: disable=broad-exception-caught
        click.echo(f"Error leyendo la imagen: {e}")

@cli.command()
@click.argument('filepath')
@click.argument('width', type=int)
@click.argument('height', type=int)
def resize(filepath, width, height):
    """Redimensiona una imagen local."""
    try:
        image = Image.open(filepath)
        new_img = resize_image(image, width, height)
        click.echo(f"Imagen redimensionada a: {new_img.size}")
    except Exception as e: # pylint: disable=broad-exception-caught
        click.echo(f"Error: {e}")

if __name__ == '__main__':
    cli()
