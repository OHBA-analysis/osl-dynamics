"""Utility classes and functions related to the models.

"""

import re


def tex_escape(text):
    """Escape characters which require control sequences in text for use
    in LaTeX.

    Parameters
    ----------
    text : str
        Text to be escaped.

    Returns
    -------
    escaped_text : str
        Escaped text.
    """

    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    regex = re.compile(
        "|".join(
            re.escape(str(key))
            for key in sorted(conv.keys(), key=lambda item: -len(item))
        )
    )
    return regex.sub(lambda match: conv[match.group()], text)


class Table:
    """A simple table class that stores data in rows and columns.

    Parameters
    ----------
    headers : list
        List of strings for the table headers.
    """

    def __init__(self, headers):
        self.headers = headers
        self.rows = []

    def __iadd__(self, other):
        """Add a row to the table.

        Parameters
        ----------
        other : list
            List of values for the row.

        Returns
        -------
        self : Table
            The table object.
        """
        if not isinstance(other, list):
            raise TypeError("Must provide a list.")
        if not len(other) == len(self.headers):
            raise ValueError("Row must have same number of elements as headers.")
        self.rows.append(other)
        return self


class HTMLTable(Table):
    """Class for creating HTML tables.

    Attributes
    ----------
    headers : list of str
        The table headers.
    rows : list of list of str
        The table rows.
    """

    outer = "<table>\n{headers}\n{content}\n</table>"
    tr = "<tr>{}</tr>"
    th = "<th>{}</th>"
    td = '<td style="white-space:pre-wrap; word-wrap:break-word">{}</td>'

    def append_last(self, item):
        """Append an item to the last cell in the last row.

        Parameters
        ----------
        item : str
            The string to append to the cell.
        """
        self.rows[-1][-1] = "\n".join([self.rows[-1][-1], item])

    def html_row(self, items):
        """Format a list of strings as an HTML table row.

        Parameters
        ----------
        items : list of str
            A list of :code:`str` to be added to the HTML table.

        Returns
        -------
        html : str
            The HTML table row.
        """
        data = "".join([self.td.format(item) for item in items])
        return self.tr.format(data)

    def html_headers(self):
        """Generate HTML table headers from self.headers

        Returns
        -------
        headers : str
            HTML table headers
        """
        headers = "".join([self.th.format(header) for header in self.headers])
        return self.tr.format(headers)

    def output(self):
        """Create the full HTML for the table.

        Returns
        -------
        html : str
            The HTML for the table.
        """
        headers = self.html_headers()
        rows = "\n".join([self.html_row(row) for row in self.rows])
        return self.outer.format(headers=headers, content=rows)

    def _repr_html_(self):
        """Return HTML representation of the current object.

        This function is for use by the Jupyter backend.

        Returns
        -------
        html : str
            HTML representation of the current object.
        """
        return self.output()


class LatexTable(Table):
    """Class for creating LaTeX tables.

    Attributes
    ----------
    headers : list of str
        The table headers.
    rows : list of list of str
        The table rows.
    vertical_lines : bool, optional
        Whether to draw vertical lines, by default :code:`True`.
    horizontal_lines : bool, optional
        Whether to draw horizontal lines, by default :code:`True`.
    header_lines : bool, optional
        Whether to draw horizontal lines above the header,
        by default :code:`True`.
    """

    outer = "".join(
        "\\begin{{centering}}\n"
        "\\resizebox{{\\textwidth}}{{!}}{{%\n"
        "\\begin{{tabular}}{{ {header_alignments} }}\n"
        "{headers} \\\\ \\hline \\hline\n"
        "{rows}\n"
        "\\end{{tabular}}\n"
        "}}\n"
        "\\end{{centering}}"
    )

    def __init__(
        self,
        headers,
        *,
        vertical_lines=True,
        horizontal_lines=True,
        header_lines=True,
    ):
        super().__init__(headers)
        self.vertical_lines = vertical_lines
        self.horizontal_lines = horizontal_lines

    def escape(self):
        """Escape special characters in headers and rows."""
        self.headers = [tex_escape(header) for header in self.headers]
        self.rows = [[tex_escape(item) for item in row] for row in self.rows]

    def append_last(self, item):
        """Append an item on the last row of the table.

        Parameters
        ----------
        item : any
            Item to append.
        """
        self.rows.append(["", "", "", item])

    @staticmethod
    def latex_row(items):
        """Create a latex row from a list of strings.

        Parameters
        ----------
        items : list of strings
            The list of strings to be turned into a row.

        Returns
        -------
        latex_row : string
            The latex row.
        """
        return " & ".join(items)

    def output(self):
        """Format the table into LaTeX code.

        Returns
        -------
        str
            The LaTeX code that produces the table.
        """
        self.escape()

        headers = self.latex_row(self.headers)

        rows = []
        for row in self.rows:
            content = self.latex_row(row)
            if row[:3] != [""] * 3:
                if self.horizontal_lines:
                    content = "\n".join(["\\hline", content])
            rows.append(content)
        rows = " \\\\ \n".join(rows)

        vertical_separator = "|" if self.vertical_lines else ""
        header_alignments = vertical_separator.join("r" for h in self.headers)
        return self.outer.format(
            headers=headers, rows=rows, header_alignments=header_alignments
        )

    def _repr_latex_(self):
        """Returns the LaTeX representation of the object.

        Returns
        -------
        latex : str
            The LaTeX representation of the object.
        """
        return self.output()
