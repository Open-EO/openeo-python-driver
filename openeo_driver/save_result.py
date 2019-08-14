from . import replace_nan_values

from abc import ABC
from typing import Dict

from flask import send_from_directory, jsonify
import os

class SaveResult(ABC):
    """
    A class that generates a Flask response.
    """

    def __init__(self,format,options):
        self.options = options
        self.format = format

    def create_flask_response(self):
        """
        Returns a Flask compatible response.

        :return: A response that can be handled by Flask
        """
        pass


class ImageCollectionResult(SaveResult):

    def __init__(self, imagecollection, format, options):
        self.imagecollection = imagecollection
        super().__init__(format,options)

    def create_flask_response(self):

        filename = self.imagecollection.download(None, bbox="", time="",format=self.format, **self.options)
        return send_from_directory(os.path.dirname(filename), os.path.basename(filename))


class JSONResult(SaveResult):

    def __init__(self, json_dict: Dict, format, options):
        self.json_dict = json_dict
        super().__init__(format, options)

    def create_flask_response(self):
        return jsonify(replace_nan_values(self.json_dict))
