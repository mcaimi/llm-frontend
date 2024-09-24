#!/usr/bin/env python
#
#
# v0.1 -- Initial Implementation -- <mcaimi@redhat.com>
#
""" Various utilities to ease dealing with console output
 from colored text to formatting to everything else """


class ANSIColors():
    """ Class managing ansi colors in console output """

    def __init__(self):
        self.escapecode = '\033[%s;%sm'

        self.colors = {'RED': 1,
                       'GREEN': 2,
                       'YELLOW': 3,
                       'BLUE': 4,
                       'PURPLE': 5,
                       'CYAN': 6,
                       'WHITE': 7,
                       'BLACK': 0}

        self.modifiers = {'BRIGHT': 1, 'NORMAL': 0}

        # build color codes hashes
        self.compile_ansicolors_hash()

    def compile_ansicolors_hash(self):
        ''' Compile ansi colors hash table '''
        colorlist = {}
        for intensity, intensity_value in self.modifiers.items():
            colorlist[intensity] = {}
            for color, color_value in self.colors.items():
                colorlist[intensity][color] = self.escapecode % (intensity_value, 30 + color_value)
        colorlist['RESET'] = '\033[0m'
        # update ansi colorlist hash
        self.ansi_escapes = colorlist

    def get_ansicolors_hash(self):
        ''' retrieve the hash table containing ANSI colors escape codes '''
        return self.ansi_escapes

    def get_code_for_color(self, mod, color):
        """ get the escape sequence with parameters

            mod: escape modifier
            color: color key """

        if mod in self.ansi_escapes:
            if color in self.ansi_escapes[mod]:
                return self.ansi_escapes[mod][color]
        return None

    def color_write(self, text, mod="NORMAL", color="WHITE"):
        ''' generate an escapecode-colored string '''
        return f"{self.get_code_for_color(mod, color)}{text}{self.ansi_escapes['RESET']}"

    def color_print(self, text, mod="NORMAL", color="WHITE"):
        ''' print colored text '''
        print(self.color_write(text, mod, color))

    def error(self, text):
        ''' generate an error string (red text) '''
        return self.color_write(text, 'BRIGHT', 'RED')

    def print_error(self, text):
        ''' print an error message (red text) '''
        print(self.error(text))

    def warning(self, text):
        ''' generate a warning string (bright yellow text) '''
        return self.color_write(text, 'BRIGHT', 'YELLOW')

    def print_warning(self, text):
        ''' print a warning message '''
        print(self.warning(text))

    def success(self, text):
        ''' generate a success string (green color) '''
        return self.color_write(text, 'BRIGHT', 'GREEN')

    def print_success(self, text):
        ''' print a success message '''
        print(self.success(text))
