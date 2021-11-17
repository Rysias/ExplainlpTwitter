from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import BarContainer


@dataclass
class NLPlotter:
    topic_titles: Dict[int, str]

    def __post_init__(self):
        self.titles = list(self.topic_titles.values())

    def plot_doc(self, doc_embed: np.ndarray) -> BarContainer:
        new_doc = self.format_embedding(doc_embed)
        fig = plt.barh(self.titles, new_doc)
        plt.xlabel("Similarity Score")

        return fig

    def plot_docs(self, source_doc: np.ndarray, target_doc: np.ndarray) -> BarContainer:
        """Plot two embeddings side to side"""
        clean_source = self.format_embedding(source_doc)
        clean_target = self.format_embedding(target_doc)

        fig, ax = plt.subplots()

        x = np.arange(len(self.titles))
        width = 0.35

        # Plotting source document
        ax.barh(x - width / 2, clean_source, width, label="Source Document")
        # Plotting target document
        ax.barh(x + width / 2, clean_target, width, label="Target Document")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel("Topic Score")
        ax.set_title("Documents compared")
        ax.set_yticks(x)
        ax.set_yticklabels(self.titles)
        ax.legend()
        return ax

    @staticmethod
    def format_embedding(embedding: np.ndarray) -> np.ndarray:
        if embedding.shape[0] == 0:
            raise ValueError("You cannot plot an empty document")
        return embedding.reshape(
            -1,
        )
