"""Utility classes and functions related to the models.

"""

import re


def tex_escape(text):
    """
    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
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
    def __init__(self, headers):
        self.headers = headers
        self.rows = []

    def __iadd__(self, other):
        if not isinstance(other, list):
            raise TypeError("Must provide a list.")
        if not len(other) == len(self.headers):
            raise ValueError("Row must have same number of elements as headers.")
        self.rows.append(other)
        return self


class HTMLTable(Table):
    outer = "<table>\n{headers}\n{content}\n</table>"
    tr = "<tr>{}</tr>"
    th = "<th>{}</th>"
    td = '<td style="white-space:pre-wrap; word-wrap:break-word">{}</td>'

    def append_last(self, item):
        self.rows[-1][-1] = "\n".join([self.rows[-1][-1], item])

    def html_row(self, items):
        data = "".join([self.td.format(item) for item in items])
        return self.tr.format(data)

    def html_headers(self):
        headers = "".join([self.th.format(header) for header in self.headers])
        return self.tr.format(headers)

    def output(self):
        headers = self.html_headers()
        rows = "\n".join([self.html_row(row) for row in self.rows])
        return self.outer.format(headers=headers, content=rows)

    def _repr_html_(self):
        return self.output()


class LatexTable(Table):
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
        self, headers, *, vertical_lines=True, horizontal_lines=True, header_lines=True
    ):
        super().__init__(headers)
        self.vertical_lines = vertical_lines
        self.horizontal_lines = horizontal_lines

    def escape(self):
        self.headers = [tex_escape(header) for header in self.headers]
        self.rows = [[tex_escape(item) for item in row] for row in self.rows]

    def append_last(self, item):
        self.rows.append(["", "", "", item])

    @staticmethod
    def latex_row(items):
        return " & ".join(items)

    def output(self):
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
        return self.output()
