#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:59:48 2023

@author: paul

Script for converting markdown cells to rst in jupyter notebooks so that 
nbsphinx can convert notebooks with cross-references to other parts of
documentation etc.

TODO: use classes to better segment parts and allow for images in tables etc.
"""

import os
import types
from pathlib import Path
import shutil
import warnings
from abc import ABC, abstractmethod
from itertools import chain
import re
import json
import numpy as np
from urllib.request import Request

import burstH2MM as bhm



def sequence_iter(*args):
    args = [iter(arg) for arg in args]
    n = len(args)
    while n != 0:
        c = 0
        while c < n:
            try:
                val = next(args[c])
            except StopIteration:
                args.pop(c)
                n -= 1
            else:
                c += 1
                yield val


def str_sub_func(func):
    if isinstance(func, str):
        return lambda _: func
    elif callable(func):
        return func
    else:
        raise ValueError(f"func must be a string or callable, got {type(func)}")
        
def match_sub(text, spans, subs):
    out_text = ''
    seq_iter = sequence_iter(spans, subs)
    beg = 0
    cont = True
    while cont:
        try:
            end, next_beg = next(seq_iter)
        except StopIteration:
            cont = False
        else:
            out_text += text[beg:end]
            beg = next_beg
        try:
            sub = next(subs)
        except StopIteration:
            cont = False
        else:
            out_text += sub.make_md()
    return out_text
        

def regex_split_iter(regex, text, subA, subB):
    riter = regex.finditer(text)
    subA, subB = str_sub_func(subA), str_sub_func(subB)
    start = 0
    for match in riter:
        end, next_start = match.span()
        yield subA(text[start:end])
        start = next_start
        yield subB(match)
    yield subA(text[start:])

    
def non_overlap(matches):
    spans = [match.span() + (i, ) for i, match in enumerate(matches)]
    sort = sorted(spans)
    if any(spanb[0] - spane[1] < 0 for spane, spanb in zip(spans[:-1], spans[1:])):
        raise ValueError("Overlaping regular expressions, cannot split")
    return sort
    

def multi_regex_split_iter(text, non_match, *args):
    args_iter = [(arg[0].finditer(text), arg[1]) for arg in args]
    matches = [None for _ in range(len(args_iter))]
    start, end, i = 0, len(text), 0
    while i < len(args_iter):
        try:
            matches[i] = next(args_iter[i][0])
        except StopIteration:
            args_iter.pop(i)
            matches.pop(i)
        else:
            i += 1
    sort = non_overlap(matches)
    while len(sort) != 0:
        stop = sort[0][0]
        if start != stop:
            yield non_match(text[start:stop])
        yield args_iter[sort[0][2]][1](matches[sort[0][2]])
        start = sort[0][1]
        try:
            matches[sort[0][2]] = next(args_iter[sort[0][2]][0])
        except StopIteration:
            args_iter.pop(sort[0][2])
            matches.pop(sort[0][2])
        sort = non_overlap(matches)
    if start != end:
        yield non_match(text[start:end])
        

def check_ref(strings):
    module = bhm
    for string in strings:
        if hasattr(module, string) or string in dir(module):
            module = getattr(module, string)
        else:
            return False
    return module

    
def attr_id(attr):
    attr_strip = attr[:-2] if attr[-2:] == '()' else attr
    strings = attr_strip.split('.')
    module = bhm
    module = check_ref(strings[:-1])
    attrib = check_ref(strings)
    if attrib == property:
        return "attr", '.'.join([module.__module__, module.__name__, strings[-1]])
    elif isinstance(attrib, types.ModuleType):
        return "mod", attrib.__name__
    elif callable(attrib) and isinstance(module, type):
        return "meth", '.'.join([module.__qualname__,attrib.__qualname__])
    elif isinstance(attrib, type):
        return "class", '.'.join([attrib.__module__, attrib.__name__])
    elif callable(attrib):
        return "func",  '.'.join([attrib.__module__, attrib.__name__])
    elif isinstance(module, type):
        return "attr", '.'.join([module.__module__, module.__name__, strings[-1]])
    else:
        return None, attr

underscore_regex = re.compile(r'\_')
notword_regex = re.compile(r'\W')
word_regex = re.compile(r'\w')

def underscore_sub(text):
    i = 1
    while i != 0:
        out_highlight, i = True, 0
        for match in underscore_regex.finditer(text):
            span = match.span()
            if out_highlight:
                if (span[0] == 0 or notword_regex.match(text[span[0]-1])) and word_regex.match(text[span[1]]):
                    loc = span[0]
                    out_highlight = False
            else:
                if word_regex.match(text[span[0]-1]) and notword_regex.match(text[span[1]]):
                    i += 1
                    text = f"{text[:loc]}*{text[loc+1:span[1]-1]}*{text[span[1]:]}"
                    out_highlight = True
    return text

leading_whitespace_regex = re.compile(r'^(\s*).*')
leading_whitespace = lambda string: len(leading_whitespace_regex.match(string).group(1))


def dedent(text):
    lines = text.split('\n')
    deden = min(leading_whitespace(line) for line in lines)
    return '\n'.join(line[deden:] for line in lines)


indent_list = lambda strings: '\n'.join('    ' + string for string in strings) + '\n'
indent = lambda string: indent_list(sub.strip() for sub in string.split('\n'))

def iter_dir(path):
    if not isinstance(path, Path):
        path = Path(path)
    for element in path.iterdir():
        if element.parts[-1][0] == '.':
            continue
        if element.is_dir():
            yield from iter_dir(element)
        yield element

def build_dir(file):
    path = Path('.')
    for part in file.parent.parts:
        path = path.joinpath(part)
        if not path.exists():
            path.mkdir()


file_ext_regex = re.compile(r'(^.+)\.(ipynb|rst)$')

def file_ext(file):
    match = file_ext_regex.match(file)
    if match:
        return match.group(2)
    else:
        return ''

def rmext(file):
    match = file_ext_regex.match(file)
    if match:
        return match.group(1)
    else:
        return ''

rstref_regex = re.compile(r'^\.\. _([^\s:]+?):\s*$', re.MULTILINE)

def get_rst_refs(file):
    with open(file, 'r') as f:
        text = '\n'.join(line for line in f)
    return [ref.group(1) for ref in rstref_regex.finditer(text)]


######################
# Defining classes
######################

class MdBase(ABC):
    #: all the refernces used
    dests = {'*docs*':list()}
    #: the directy where the documention is accumulated
    docs_dir = list()
    # list of notebook files analyzed
    notebooks = list()
    #: base directory of notebooks
    nb_dir = None
    #: the current notebook
    curr_file = None
    #: current cell number
    curr_cell = None
    #: location in readthedocs.io
    base_url = None
    #: files that need to be copied
    files = list()
    @classmethod
    def set_base_url(cls, url):
        cls.base_url = url
    @classmethod
    def new_notebook(cls, notebook):
        if notebook not in cls.dests:
            cls.dests['*docs*'].append(str(notebook.relative_to(cls.nb_dir))[:-6])
            cls.dests[notebook] = list()
            cls.notebooks.append(notebook)
            cls.curr_file = notebook
        else:
            raise ValueError(f"{notebook} already processed")
    @classmethod
    def set_notebook(cls, notebook):
        if notebook in cls.dests:
            cls.curr_file = notebook
        else:
            raise KeyError(f"{notebook} not yet processed")
    @classmethod
    def ref_dir(cls, source_dir):
        cls.dests.update({file.relative_to(source_dir):get_rst_refs(file) 
                                                   for file in iter_dir(source_dir) 
                                                   if file.parts[-1][-4:] == '.rst'})
        cls.dests['*docs*'] += [str(file.relative_to(source_dir))[:-4] 
                                    for file in iter_dir(source_dir) 
                                    if file.parts[-1][-4:]=='.rst']
        cls.docs_dir = source_dir
    @classmethod
    def copy_files(cls, res_dir):
        for file in cls.files:
            nfile = res_dir.joinpath(file.relative_to(cls.nb_dir))
            build_dir(nfile)
            shutil.copy2(file, nfile)
    def __len__(self):
        return len(self.make())
    @abstractmethod
    def make(self):
        pass
    @property
    def span(self):
        return max(len(string) for string in self.make().split('\n'))
    @property
    def rows(self):
        return self.make().count('\n')
    @property
    @abstractmethod
    def begline(self):
        pass
    @property
    @abstractmethod
    def endline(self):
        pass
    
class MdContents(MdBase):
    def make(self):
        return '.. contents::'
    @property
    def begline(self):
        return 2
    @property
    def endline(self):
        return 2
    


class MdSplitLine(MdBase):
    def make(self):
        return ''
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 2


class MdCodeBlock(MdBase):
    def __init__(self, text):
        code_type, code = text.split('\n', 1)
        self.code_type = code_type
        self.code = code
    def make(self):
        return f'.. code:: {self.code_type}\n\n' + indent(self.code)
    @property
    def begline(self):
        return 2
    @property
    def endline(self):
        return 2
 
    
begline_regex = re.compile('^\s*?\n')
    
class MdUtext(MdBase):
    def __init__(self, text):
        self.text = text
    def make(self):
        return self.text
    @property
    def begline(self):
        return 1 if bool(begline_regex.match(self.text)) else 0
    @property
    def endline(self):
        return 1 if self.text[-1] == '\n' else 0
    def split(self, n=1):
        return (MdUtext(txt) for txt in self.text.split('\n', n))

class MdText(MdBase):
    def __init__(self, text):
        text = dedent(underscore_sub(text))
        self.text = text
    def make(self):
        return self.text
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0

class MdAtomic(MdBase):
    pass


class MdDest(MdAtomic):
    def __init__(self, name):
        self.name = name
        self.dests[self.curr_file].append(name)
    def make(self):
        return f'.. _{self.name}:'
    @property
    def begline(self):
        return 1
    @property
    def endline(self):
        return 2
        


def mdMathAttr(match):
    sort, name = match.group(1), match.group(2)
    if sort =='$':
        return MdMathSingle(name)
    elif sort == '$$':
        return MdMathDouble(name)
    else:
        kind, attr_name = attr_id(name)
        if kind is None:
            return MdCodeSnip(name)
        else:
            return MdAttr(name, kind, attr_name)
        

class MdMathSingle(MdAtomic):
    def __init__(self, equation):
        self.equation = ' '.join(equation.split('\n'))
    def make(self):
        return f':math:`{self.equation}`'
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0


class MdMathDouble(MdAtomic):
    def __init__(self, equation):
        self.equation = equation
    def make(self):
        return '.. math::\n\n' + indent(self.equation)
    @property
    def begline(self):
        return 2
    @property
    def endline(self):
        return 2


class MdAttr(MdAtomic):
    def __init__(self, name, kind, attr_name):
        self.name = name
        self.kind = kind
        self.attr_name = attr_name
    def make(self):
        return f':{self.kind}:`{self.name} <{self.attr_name}>`'
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0


class MdCodeSnip(MdAtomic):
    def __init__(self, code, *args):
        self.code = code
    def make(self):
        return f'``{self.code}``'
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0


class MdSupSub(MdAtomic):
    def __init__(self, group, text):
        self.text = underscore_sub(text.replace('\n', ' '))
        self.group = group
    def make(self):
        return f':{self.group}:`{self.text}`'
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0


class MdLink(MdAtomic):
    _registry = {}
    def __init_subclass__(cls, kind, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[kind] = cls   
    def __new__(cls, match):
        kind, name, ref = cls._get_kind(match)
        subclass = cls._registry[kind]
        obj = object.__new__(subclass)
        obj.name = name.replace('\n', ' ')
        obj.ref = ref
        return obj
    @staticmethod
    def _get_kind(match):
        fig, name, ref = match.group(1) =='!', match.group(2), match.group(3)
        if fig:
            kind = 'img'
        elif ref[0] == '#':
            ref = ref[1:]
            kind = 'ilink'
        elif Path(ref).is_file():
            kind = 'down'
        else:
            kind = 'url'
        return kind, name, ref
    @classmethod
    def gen_link(cls, match):
        kind, name, ref = cls._get_kind(match)
        if kind == 'url':
            return f'[{name}]({ref})'
        elif kind in ('img', 'down'):
            subdir = str(cls.curr_file.parent.relative_to(cls.nb_dir))
            if subdir == '.':
                subdir = ''
            url = cls.base_url + '_' + subdir + ref
            link = f'[{name}]({url})'
            if kind == 'img':
                link = '!' + link
            return link
        elif kind == 'ilink':
            if ref in cls.dests[cls.curr_file]:
                return f'[{name}](#{ref})'
            else:
                for file, loc in cls.dests.items():
                    if ref in loc:
                        if isinstance(file, str):
                            return f'[{name}]({cls.base_url}{ref}.html)'
                        else:
                            fstr = rmext((str(file)))
                            return f'{cls.base_url}#{fstr}.html{ref}'
                print(f'reference {ref} does not exist in files')
                warnings.warn(f'reference {ref} does not exist in files')
                return match.group()
   

class MdIlink(MdLink, kind='ilink'):
    def __init__(self, match):
        if self.ref in self.dests['*docs*']:
            self.type = 'doc'
        else:
            self.type = 'ref'
    def make(self):
        return f' :{self.type}:`{self.name} <{self.ref}_>`'
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0
    

class MdFigure(MdLink, kind='img'):
    def __init__(self, match):
        file = self.curr_file.parent.joinpath(Path(self.ref))
        if not os.path.isfile(file):
            raise FileNotFoundError(f"{file} does not exist")
        self.files.append(file)
        self.dest = None
    def make(self):
        text = f'.. figure:: {self.ref}\n:alt: {self.name}'
        if hasattr(self, "caption"):
            text += f'\n\n    {"".join(comp.make() for comp in self.caption)}'
        if self.dest is not None:
            text = self.dest.make() + '\n\n' + text
        return text
    @property
    def begline(self):
        return 2
    @property
    def endline(self):
        return 2
    def add_caption(self, caption):
        if isinstance(caption, str):
            self.caption = (MdText(caption), )
        else:
            ncaption = list()
            for atom in caption:
                if isinstance(atom, str):
                    ncaption.append(MdText(atom))
                elif isinstance(atom, MdUtext):
                    ncaption.append(MdText(atom.text))
                elif isinstance(atom, MdDest):
                    self.dest = atom
                elif isinstance(atom, MdAtomic):
                    ncaption.append(atom)
            self.caption = tuple(ncaption)
    

class MdDownload(MdLink, kind='down'):
    def __init__(self, match):
        file = self.curr_file.parent.joinpath(Path(self.ref))
        if not os.path.isfile(file):
            raise FileNotFoundError(f"{file} does not exist")
        self.files.append(file)
    def make(self):
        return f':download:`{self.name} <{self.ref}>`'
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0
    

class MdUrl(MdLink, kind='url'):
    def __init__(self, match):
        try:
            Request(self.ref)
        except ValueError as e:
            raise ValueError(f'Unknown url type in link: [{self.name}]({self.ref})') from e
    def make(self):
        return f' `{self.name} <{self.ref}>`_'
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0
    

class MdHighLevel(MdBase):
    @staticmethod
    @abstractmethod
    def can_keep(*args):
        pass
    
class MdHeading(MdHighLevel):
    levels = ('#', '*', '=', '-', )
    @staticmethod
    def can_keep(atom):
        if isinstance(atom, MdAtomic) and atom.begline == 0 and atom.endline == 0:
            return (0, atom)
        elif isinstance(atom, MdUtext):
            return (1, ) + tuple(atom.split())
        else:
            return (-1, atom)
    def __init__(self, match, *args):
        level, text = match.group(1), match.group(2)
        self.elements = [MdText(text), ] + list(args)
        self.level = len(level) - 1
    def make(self):
        heading = ''.join(element.make() for element in self.elements)
        return heading + '\n' + MdHeading.levels[self.level]*len(heading)
    @property
    def begline(self):
        return 2
    @property
    def endline(self):
        return 2
    @property
    def rows(self):
        return 2

note_strip_regex = re.compile(r'^(\s*>)(.*)', re.MULTILINE)
note_strip_sub = lambda match: match.group(2)
note_strip = lambda match: note_strip_regex.sub(note_strip_sub, match.group())

note_label_regex = re.compile(r'^\s*(\*\*|\_\_)([Nn]ote|[Ss]ee [Aa]lso|[Ss]ee[Aa]lso|[Ww]arning)\1s*(.*)', 
                              re.DOTALL)


class MdNote(MdHighLevel):
    @staticmethod
    def can_keep(atom):
        if isinstance(atom, (MdAtomic, MdUtext)):
            return (0, atom)
        else:
            return (1, atom)
    def __init__(self, match, *args):
        start_text = note_strip(match)
        label = note_label_regex.match(start_text)
        if label:
            self.label = label.group(2).lower()
            start_text = label.group(3).split('\n',1)
            if len(start_text) == 2:
                start_text = start_text[0] + '\n' + dedent(start_text[1].strip('\n'))
            else:
                start_text = start_text[0]
        else:
            self.label = None
            start_text = dedent(start_text)
        self.elements = procHighlevel([MdUtext(start_text), ] + list(args))
        
    def make(self):
        if self.label is None:
            return indent(render_RST(self.elements))
        else:
            return f'.. {self.label}::\n\n' + indent(f'some paragraph {render_RST(self.elements)}')
    @property
    def begline(self):
        return 2
    @property
    def endline(self):
        return 2

table_possible_regex = re.compile(r'(.*\|.*\n)+')

cell_split_len = lambda string: max(len(split) for split in string.split(' '))

def cell_split_iter(components):
    for comp in components:
        if isinstance(comp, str):
            yield from ((False, s) for s in comp.split(' '))
        else:
            yield (comp.endline or comp.begline), comp

def cell_min_span(cell):
    if len(cell) == 0:
        return 0
    else:
        return max(elem.span if isinstance(elem, MdBase) else cell_split_len(elem) for elem in cell)

def replace_Dest(row, dests):
    for i, cell in enumerate(row):
        for j, atom in enumerate(cell):
            if isinstance(atom, MdDest):
                row[i][j] = ''
                dests.append(atom)

def pad_cell(text, span, rows):
    text += '\n' * (rows - text.count('\n') - 1)
    for row in text.split('\n'):
        yield ' ' + row + ' '*(span-len(row) + 1)

sjoin = lambda s, siter: s + s.join(siter) + s
    
class MdTable(MdHighLevel):
    @staticmethod
    def can_keep(atom):
        if not isinstance(atom, (MdAtomic, MdUtext)):
            return (-1, atom)
        elif isinstance(atom, MdAtomic):
            return (0, atom)
        elif isinstance(atom, MdUtext):
            tbl_split = table_possible_regex.split(MdUtext.text)
            return (1, ) + tuple(tbl_split)
        else:
            return (-1, atom)
            
    def __init__(self, header, align, *rows):
        self.dests = list()
        body = [header, ] + list(rows)
        for row in body:
            replace_Dest(row, self.dests)
        self.cols = len(header)
        cells = [[MdCell(cell) for i, cell in enumerate(row) if i < self.cols] for row in body]
        for cell in cells:
            cell += [MdCell(["", ]) for i in range(self.cols - len(cell))]
        self.align = align
        self.cells = cells
    def make(self):
        col_span = tuple(max(cell.min_span for cell in (row[i] for row in self.cells)) 
                         for i in range(self.cols))
        col_div = sjoin("+", ("-"*(cspan+2) for cspan in col_span))
        if self.trows != 2:
            table_divs = (sjoin("+", ("="*(cspan+2) for cspan in col_span)), )
        else:
            table_divs = (col_div, )
        table_divs += tuple(col_div for _ in range(self.trows -1))
        text = col_div + ''
        cell_lines = ([cell.make(cspan) for cspan, cell in zip(col_span, cells)] for cells in self.cells)
        for lines, divs in zip(cell_lines, table_divs):
            max_row = max(cell.count('\n') for cell in lines) + 1
            line_iter = zip(*(pad_cell(cell, cspan, max_row) for cell, cspan in zip(lines, col_span)))
            text += sjoin('\n', (sjoin('|', line) for line in line_iter))
            text += divs
        return text
    @property
    def begline(self):
        return 1
    @property
    def endline(self):
        return 1
    @property
    def trows(self):
        return len(self.cells)
    @property
    def tcols(self):
        return len(self.cells[0])


whitespace_regex = re.compile(r'^\s*$')

        
class MdCell(MdBase):
    _min_span = 16
    def __init__(self, components):
        comps = list(components)
        if len(comps) > 1 and isinstance(comps[-1], str) and whitespace_regex.match(comps[-1]):
            comps.pop()
        if len(comps) > 1 and isinstance(comps[0], str) and whitespace_regex.match(comps[0]):
            comps.pop(0)
        if isinstance(comps[0], str):
            comps[0] = comps[0].lstrip(' ')
        if isinstance(comps[-1], str):
            comps[-1] = comps[-1].rstrip(' ')
        self.components = comps
    def make(self, span=np.inf):
        prev_end = 0
        cur_width = 0
        text = ''
        for elem in self.components:
            if isinstance(elem, str):
                lines = elem.split('\n')
                for line in lines:
                    if len(line) + cur_width <= span:
                        text += line
                    else:
                        for word in line.split(' '):
                            if cur_width + 1 + len(word) <= span:
                                text += word + ' ' 
                                cur_width += 1 + len(word)
                            else:
                                if word != '':
                                    text += '\n' + word + ' '
                                    cur_width = len(word) + 1
                                else:
                                    text += '\n'
                                    cur_width = 0
            else:
                string = elem.make()
                begline, endline = elem.begline, elem.endline
                if endline == begline == 0:
                    swidth = len(string)
                    if swidth + cur_width <= span:
                        text += string
                        cur_width += swidth
                    else:
                        text += '\n' + string
                        cur_width = swidth
                else:
                    text += '\n'*(begline - prev_end) + string + '\n'*endline
                    prev_end = endline
                    cur_width = 0
        return text
        
    @property
    def min_span(self):
        min_span = cell_min_span(self.components)
        if len(self.components) == 1 and isinstance(self.components[0], str) and len(self.components[0]) < self._min_span:
            return len(self.components[0])
        else:
            return min_span if min_span > self._min_span else self._min_span
    @property
    def begline(self):
        return 0
    @property
    def endline(self):
        return 0


#######################################
# Text processing iterable functions
#######################################

bar_regex = re.compile(r'\|')
table_hcell_regex = re.compile(r'\s*[:\-]{3}[:\-\s]*')
table_hmcell_regex = re.compile(r'[:\-\s]*')
escape_bar_regex = re.compile(r'\\\|')
table_line_count = lambda line: line.count('|') - len(escape_bar_regex.split(line)) + 1

def allproc(group):
    for element in group:
        if isinstance(element, MdUtext):
            yield MdText(element.text)
        else:
            yield element

            
def check_figtable(table_lines):
    csize = len(table_lines) == 3 and len(table_lines[0]) == 1
    # ensure all components of first cell are figures or empty whitespace
    hasfig = all(isinstance(cell, MdFigure) or (isinstance(cell, str) and whitespace_regex.match(cell)) 
                 for cell in table_lines[0][0])
    # next ensure only one figure if first cell
    numfig = sum(isinstance(cell, MdFigure) for cell in table_lines[0][0]) == 1
    # finally ensure caption contains no breaking elements
    nobreak = all(isinstance(cell, str) or (isinstance(cell, MdAtomic) 
                                             and cell.endline==0 and cell.begline==0)
                      for cell in table_lines[2][0])
    if csize and hasfig and numfig and nobreak:
        for cell in table_lines[0][0]:
            if isinstance(cell, MdFigure):
                fig = cell
                break
        fig.add_caption(table_lines[2][0])
        return fig
    else:
        return MdTable(*table_lines)
    

def table_line_split(line):
    if line == '':
        return [['',],]
    cells = list()
    start = 0
    for match in bar_regex.finditer(line):
        loc, locn = match.span()
        if loc != 0 and line[loc-1] != '\\':
            cells.append([line[start:loc], ])
            start = locn
        elif loc == 0:
            start = 1
    if start != len(line):
        cells.append([line[start:], ])
    return cells


def testTable(table_list):
    table_lines = [[list(), ], ]
    cell_count = cell_count = [0, ]
    not_check_align = True
    for element in table_list:
        if isinstance(element, MdAtomic):
            table_lines[-1][-1].append(element)
        else:
            lines = element.text.split('\n')
            # special case of first-line-first-cell, must append to list of previous cell list
            line = lines[0]
            cell_count[-1] += table_line_count(line)
            cells = table_line_split(line)
            table_lines[-1][-1].append(cells[0][0])
            # append rest of line
            for cell in cells[1:]:
                table_lines[-1].append(cell)
            if len(table_lines[-1][0]) == 1 and isinstance(table_lines[-1][0][0], str) and table_lines[-1][0][0].strip() == '':
                table_lines[-1].pop(0)
            # append the remaining lines
            for line in lines[1:]:
                cell_count.append(table_line_count(line))
                cells = table_line_split(line)
                table_lines.append(cells)
            if not_check_align and len(lines) > 1:
                not_check_align = False
                vfirst =  table_hcell_regex.match(table_lines[1][0][0])
                vmid = all(table_hmcell_regex.match(cell[0]) for cell in table_lines[1][1:])
                colmatch = len(table_lines[0]) == len(table_lines[1])
                if not (vfirst and vmid and colmatch):
                    yield from allproc(table_list)
                    return StopIteration
    if len(table_lines) >= 2:
        yield check_figtable(table_lines)
    else:
        yield from allproc(table_list)

def locate_table(text, siter):
    if table_line_count(text.text.split('\n')[0]) == 0 and text.text.count('\n') != 0:
        yield MdText(text.text)
        return StopIteration
    group = [text]
    siter = iter(siter)
    for ntext in siter:
        if not isinstance(ntext, (MdAtomic, MdUtext)):
            yield from testTable(group)
            yield ntext
            return StopIteration
        group.append(ntext)
    yield from testTable(group)


heading_regex = re.compile(r'^[^\S\n]*?(#+)[^\S\n]+(.+)', re.MULTILINE)
note_regex = re.compile(r'^[^\S\r\n]*>(([^\S\n]*?[^\s\#].*\n?)+)', re.MULTILINE)


dest_regex = re.compile(r"""<a id=['"](.+?)['"]></a>""")
dest_sub = lambda match: (MdDest(match.group(1)), )

dest_gen = lambda text: (MdUtext(text), )

ref_regex = re.compile(r'(\!)?\[(.+?)\]\((.+?)\)', re.MULTILINE|re.DOTALL)
ref_sub = lambda match: (MdLink(match), )

ref_gen = lambda text: chain.from_iterable(regex_split_iter(dest_regex, text, dest_gen, dest_sub))


supsub_regex = re.compile(r'<(su[pb])>(.*?)</\1>')
supsub_sub = lambda match: (MdSupSub(match.group(1), match.group(2)), )

supsub_gen = lambda text: chain.from_iterable(regex_split_iter(ref_regex, text, ref_gen, ref_sub))


mathattr_regex = re.compile(r'(\$\$|\$|``|`)(.*?[^\\])\1', re.DOTALL|re.MULTILINE)
mathattr_sub = lambda match: (mdMathAttr(match), )

mathattr_gen = lambda text: chain.from_iterable(regex_split_iter(supsub_regex, text, supsub_gen, supsub_sub))



doubleline_regex = re.compile(r'\n[^\S\n]*\n')
doubleline_sub = lambda text: (MdSplitLine(),)

doubleline_gen = lambda text: chain.from_iterable(regex_split_iter(mathattr_regex, text, 
                                                   mathattr_gen, mathattr_sub))


codeblock_regex = re.compile(r'```(.*)```', re.DOTALL)
codeblock_sub = lambda match: (MdCodeBlock(match.group(1)), )


codeblock_gen = lambda text: chain.from_iterable(regex_split_iter(doubleline_regex, text, 
                                                 doubleline_gen, doubleline_sub))



base_gen = lambda text: chain.from_iterable(regex_split_iter(codeblock_regex, 
                                                             text, codeblock_gen, 
                                                             codeblock_sub))


eUtext = lambda text: MdUtext(text)


heading_sub = lambda match: (MdHeading, match)

note_sub = lambda match: (MdNote, match)

echo_text = lambda text: (text, )

echo_match = lambda match: (match.group(), )


def md_ref_sub(match):
    return (MdLink.gen_link(match), )

def md_mathattr_sub(match):
    sort, name = match.group(1), match.group(2)
    if '`' not in sort:
        return (match.group(), )
    else:
        kind, attr_name = attr_id(name)
        if kind is None:
            return (match.group(), )
        else:
            return f'[{name}]({MdBase.base_url}Documentation.html#{attr_name})'


md_ref_gen = lambda text: chain.from_iterable(regex_split_iter(ref_regex, text, echo_text, md_ref_sub))

md_subsub_gen = lambda text: chain.from_iterable(regex_split_iter(supsub_regex, text, md_ref_gen, echo_match))

md_mathattr_gen = lambda text: chain.from_iterable(regex_split_iter(mathattr_regex, text, 
                                                                    md_subsub_gen, 
                                                                    md_mathattr_sub))

md_doubleline_gen = lambda text: chain.from_iterable(regex_split_iter(doubleline_regex, text, 
                                                                      md_mathattr_gen, 
                                                                      echo_match))

md_base_gen = lambda text: ''.join(chain.from_iterable(regex_split_iter(codeblock_regex, text, 
                                                                        md_doubleline_gen, 
                                                                        echo_match)))


def combineHighlevel(elements, mdClass, init):
    inputs = [init, ]
    keep = (0, )
    while keep[0] == 0:
        try:
            atom = next(elements)
        except StopIteration:
            keep = (1, )
            yield mdClass(*inputs)
        else:
            keep = mdClass.can_keep(atom)
            pos, out = keep[0], keep[1:]
            if pos == 0:
                inputs += out
            elif pos > 0:
                inputs += out[:pos]
                yield mdClass(*inputs)
                for outy in out[pos:]:
                    yield outy
            else:
                inputs += out[-pos:]
                yield mdClass(*inputs)
                for outy in out[:-pos]:
                    yield outy

def procUtext(element, element_iter):
    headnote_iter = multi_regex_split_iter(element.text, eUtext, 
                                           (note_regex, note_sub), 
                                           (heading_regex, heading_sub))
    try:
        comp_prev = next(headnote_iter)
    except StopIteration:
        return StopIteration
    else:
        for comp_new in headnote_iter:
            if isinstance(comp_prev, MdUtext):
                yield from locate_table(comp_prev, headnote_iter)
#                 yield comp_prev
            else:
                yield comp_prev[0](comp_prev[1])
            comp_prev = comp_new
        if isinstance(comp_prev, MdUtext):
            yield from locate_table(comp_prev, element_iter)
#             yield comp_prev  
        else:
            # must yield from so that can yield either just Highlevel, or highlevel and next
            yield from combineHighlevel(element_iter, *comp_prev)

def procHighlevel(elements):
    element_iter = iter(elements)
    end_iter = False
    for element in element_iter:
        # iterate until reach uncprocessed text
        while not isinstance(element, (MdUtext)):
            yield element
            try:
                element = next(element_iter)
            except StopIteration:
                end_iter = True
                break
        # either reached the end (end_iter=True) or reached MdUtext
        if end_iter:
            break
        yield from procUtext(element, element_iter)

def procHighlevel_contents(elements):
    highlevel = procHighlevel(elements)
    try:
        first_row = next(highlevel)
    except StopIteration:
        return StopIteration
    yield first_row
    if isinstance(first_row, MdHeading):
        yield MdContents()
    yield from highlevel
        


endline_count_regex = re.compile(r'.*?(\n*)$', re.DOTALL)
endline_count = lambda text: len(endline_count_regex.match(text).group(1))

def render_RST(elements):
    text = ''
    prev_end = 2
    for elem in elements:
        cur_beg, endline = elem.begline, elem.endline
        beg_line = cur_beg - prev_end - 1
        text += '\n'*beg_line + elem.make() + '\n'*endline
        prev_end = endline
    return text


def init_nb(file, nb_dir, out_dir):
    out_file = out_dir.joinpath(file.relative_to(nb_dir))
    with open(file, 'r') as nb:
        orig_nb = json.load(nb)
    out_nb = dict()
    for key, val in orig_nb.items():
        if key == 'cells':
            out_nb[key] = list()
        else:
            out_nb[key] = val
    return out_file, orig_nb, out_nb


def convert_notebook(file, nb_dir, rst_dir, md_dir):
    MdBase.new_notebook(file)
    out_file, orig_nb, rst_nb = init_nb(file, nb_dir, rst_dir)
    cells = enumerate(orig_nb['cells'])
    i, cell = next(cells)
    if cell['cell_type'] != 'markdown':
        raise ValueError("First cell must be markdown for conversion")
    source = ''.join(cell['source'])
    text_split = [text + '\n' for text in render_RST(procHighlevel_contents(base_gen(source))).split('\n')]
    text_split[-1] = text_split[-1][:-1]
    rst_nb['cells'].append({'cell_type':'raw', 'metadata': {'raw_mimetype': 'text/restructuredtext'},
                                   'id':cell['id'], 'source':text_split})
    for i, cell in cells:
        if cell['cell_type'] != 'markdown':
            rst_nb['cells'].append(cell)
            continue
        source = ''.join(cell['source'])
        text_split = [text + '\n' for text in render_RST(procHighlevel(base_gen(source))).split('\n')]
        text_split[-1] = text_split[-1][:-1]
        rst_nb['cells'].append({'cell_type':'raw', 'metadata': {'raw_mimetype': 'text/restructuredtext'},
                                       'id':cell['id'], 'source':text_split})
    if rst_nb['cells'][-1]['cell_type'] == 'raw':
        md_file = str(md_dir.joinpath(file.relative_to(nb_dir)).relative_to(rst_dir))
        download_text = ['\n', '\n', f'Download this documentation as a jupyter notebook here: :download:`{file.parts[-1]} <{md_file}>`']
        rst_nb['cells'][-1]['source'] += download_text
    with open(out_file, 'w') as nb:
        json.dump(rst_nb, nb)

def change_md_refs(file, nb_dir, md_dir):
    out_file, orig_nb, md_nb = init_nb(file, nb_dir, md_dir)
    MdBase.set_notebook(file)
    for i, cell in enumerate((orig_nb['cells'])):
        if cell['cell_type'] != 'markdown':
            md_nb['cells'].append(cell)
            continue
        text = ''.join(cell['source'])
        text_sub = md_base_gen(text)
        split_text = [txt + '\n' for txt in text_sub.split('\n')]
        split_text[-1] = split_text[-1][:-1]
        cell['source'] = split_text
        md_nb['cells'].append(cell)
    build_dir(out_file)
    with open(out_file, 'w') as nb:
       json.dump(md_nb, nb)


if __name__ == "__main__":
    # global files_list
    # files_list = list()
    # specify directories (convert to cli arguments in future) 
    nb_dir = Path('notebooks')
    source_dir = Path('source')
    md_dir = Path('test_convert/notebooks')
    rst_dir = Path('test_convert')
    # convert all notebooks
    MdBase.base_url = 'https://bursth2mm.readthedocs.io/en/latest/'
    MdBase.nb_dir = nb_dir
    MdBase.ref_dir(source_dir)
    for file in iter_dir(MdBase.nb_dir):
        if file.parts[-1][-6:] != '.ipynb':
            continue
        convert_notebook(file, nb_dir, rst_dir, md_dir)
    MdBase.copy_files(rst_dir)
    for file in MdBase.notebooks:
        if isinstance(file, str):
            continue
        change_md_refs(file, nb_dir, md_dir)
        
    # get all other references
    # for filename in os.listdir('source/'):
    #     if filename[-4:] != '.rst':
    #         continue
    #     with open(os.path.join('source', filename), 'r') as file:
    #         dest_list += [rst_dest_regex.match(line).group(1) for line in file if rst_dest_regex.match(line)]
    # for filename in files_list:
    #     filepath = Path(os.path.relpath(filename)).parts[1:]
    #     newpath = Path('test_convert')
    #     for dr in filepath[:-1]:
    #         newpath = os.path.join(newpath, dr)
    #         if not os.path.isdir(newpath):
    #             os.mkdir(newpath)
    #         shutil.copy2(filename, os.path.join(newpath, filepath[-1]))
        