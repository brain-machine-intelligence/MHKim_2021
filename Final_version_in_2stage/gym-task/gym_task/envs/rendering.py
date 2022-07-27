"""
2D rendering framework
"""
import os
import sys

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym import error

try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL installed. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )

import math
import numpy as np

RAD2DEG = 57.29577951308232


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[1].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        fullscreen=True,
        **kwargs
    )


class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = get_window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        self.window.flip()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]

    def __del__(self):
        self.close()


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(
            self.translation[0], self.translation[1], 0
        )  # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self):
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


class Point(Geom):
    def __init__(self):
        Geom.__init__(self)

    def render1(self):
        glBegin(GL_POINTS)  # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        # glColor3f(0.0,0.0,1.0)
        glEnd()


def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(
            -self.width / 2, -self.height / 2, width=self.width, height=self.height
        )


# ================================================================

# 'C:/Users/kmh/Documents/Atari_add/MH/resource/comp_311I.png'

class SimpleImageViewer(object):
    def __init__(self, width, height, prev, seq, game, order, display=None, maxwidth=1600):

        self.window = None
        self.isopen = False
        self.display = get_display(display)
        self.maxwidth = maxwidth
        self.width = width
        self.height = height

        self.label = None
        self.label_tot = None

        # current complexity
        self.comp = seq[0]

        # game type
        self.game = game

        # 연산 있는 complexity의 경우 order가 1이면 L or U or O. 2이면 R or D or I 의미
        self.order = order

        # previous reward and current reward
        self.prev = prev
        self.cur = 0

        # circle image dictionary
        self.circles = self.load_images()

        # sequence of complexity given as input
        self.sequence = seq

    # load circles from the proper folder - need to be changed
    def load_images(self):

        # {complexity : 이미지 파일}로 저장. complexity + "1" 또는 complexity + "2" 통해 첫번째 항인지 두번째 항인지 구분

        image_dic = {}
        image_dir = 'C:/Users/kmh/Documents/Atari_add/MH/resource/'
        image_dic.update({"0": pyglet.image.load(image_dir + 'comp_0.png')})
        image_dic.update({"1": pyglet.image.load(image_dir +'comp_1.png')})

        image_dic.update({"100": pyglet.image.load(image_dir + 'comp_100.png')})
        image_dic.update({"1001": pyglet.image.load(image_dir + 'comp_100L.png')})
        image_dic.update({"1002": pyglet.image.load(image_dir + 'comp_100R.png')})

        image_dic.update({"101": pyglet.image.load(image_dir + 'comp_101.png')})
        image_dic.update({"1011": pyglet.image.load(image_dir + 'comp_101L.png')})
        image_dic.update({"1012": pyglet.image.load(image_dir + 'comp_101R.png')})

        image_dic.update({"110": pyglet.image.load(image_dir + 'comp_110.png')})
        image_dic.update({"1101": pyglet.image.load(image_dir + 'comp_110L.png')})
        image_dic.update({"1102": pyglet.image.load(image_dir + 'comp_110R.png')})

        image_dic.update({"111": pyglet.image.load(image_dir + 'comp_111.png')})
        image_dic.update({"1111": pyglet.image.load(image_dir + 'comp_111L.png')})
        image_dic.update({"1112": pyglet.image.load(image_dir + 'comp_111R.png')})

        image_dic.update({"200": pyglet.image.load(image_dir + 'comp_200.png')})
        image_dic.update({"2001": pyglet.image.load(image_dir + 'comp_200U.png')})
        image_dic.update({"2002": pyglet.image.load(image_dir + 'comp_200D.png')})

        image_dic.update({"201": pyglet.image.load(image_dir + 'comp_201.png')})
        image_dic.update({"2011": pyglet.image.load(image_dir + 'comp_201U.png')})
        image_dic.update({"2012": pyglet.image.load(image_dir + 'comp_201D.png')})

        image_dic.update({"210": pyglet.image.load(image_dir + 'comp_210.png')})
        image_dic.update({"2101": pyglet.image.load(image_dir + 'comp_210U.png')})
        image_dic.update({"2102": pyglet.image.load(image_dir + 'comp_210D.png')})

        image_dic.update({"211": pyglet.image.load(image_dir + 'comp_211.png')})
        image_dic.update({"2111": pyglet.image.load(image_dir + 'comp_211U.png')})
        image_dic.update({"2112": pyglet.image.load(image_dir + 'comp_211D.png')})

        image_dic.update({"300": pyglet.image.load(image_dir + 'comp_300.png')})
        image_dic.update({"3001": pyglet.image.load(image_dir + 'comp_300O.png')})
        image_dic.update({"3002": pyglet.image.load(image_dir + 'comp_300I.png')})

        image_dic.update({"301": pyglet.image.load(image_dir + 'comp_301.png')})
        image_dic.update({"3011": pyglet.image.load(image_dir + 'comp_301O.png')})
        image_dic.update({"3012": pyglet.image.load(image_dir + 'comp_301I.png')})

        image_dic.update({"310": pyglet.image.load(image_dir + 'comp_310.png')})
        image_dic.update({"3101": pyglet.image.load(image_dir + 'comp_310O.png')})
        image_dic.update({"3102": pyglet.image.load(image_dir + 'comp_310I.png')})

        image_dic.update({"311": pyglet.image.load(image_dir + 'comp_311.png')})
        image_dic.update({"3111": pyglet.image.load(image_dir + 'comp_311O.png')})
        image_dic.update({"3112": pyglet.image.load(image_dir + 'comp_311I.png')})


        # image_dic.update({"0": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_0.png')})
        # image_dic.update({"1": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_1.png')})
        #
        # image_dic.update({"100": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_100.png')})
        # image_dic.update({"1001": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_100L.png')})
        # image_dic.update({"1002": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_100R.png')})
        #
        # image_dic.update({"101": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_101.png')})
        # image_dic.update({"1011": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_101L.png')})
        # image_dic.update({"1012": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_101R.png')})
        #
        # image_dic.update({"110": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_110.png')})
        # image_dic.update({"1101": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_110L.png')})
        # image_dic.update({"1102": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_110R.png')})
        #
        # image_dic.update({"111": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_111.png')})
        # image_dic.update({"1111": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_111L.png')})
        # image_dic.update({"1112": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_111R.png')})
        #
        # image_dic.update({"200": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_200.png')})
        # image_dic.update({"2001": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_200U.png')})
        # image_dic.update({"2002": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_200D.png')})
        #
        # image_dic.update({"201": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_201.png')})
        # image_dic.update({"2011": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_201U.png')})
        # image_dic.update({"2012": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_201D.png')})
        #
        # image_dic.update({"210": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_210.png')})
        # image_dic.update({"2101": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_210U.png')})
        # image_dic.update({"2102": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_210D.png')})
        #
        # image_dic.update({"211": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_211.png')})
        # image_dic.update({"2111": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_211U.png')})
        # image_dic.update({"2112": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_211D.png')})
        #
        # image_dic.update({"300": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_300.png')})
        # image_dic.update({"3001": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_300O.png')})
        # image_dic.update({"3002": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_300I.png')})
        #
        # image_dic.update({"301": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_301.png')})
        # image_dic.update({"3011": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_301O.png')})
        # image_dic.update({"3012": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_301I.png')})
        #
        # image_dic.update({"310": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_310.png')})
        # image_dic.update({"3101": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_310O.png')})
        # image_dic.update({"3102": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_310I.png')})
        #
        # image_dic.update({"311": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_311.png')})
        # image_dic.update({"3111": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_311O.png')})
        # image_dic.update({"3112": pyglet.image.load('C:/Users/kmh/Documents/Atari_add/MH/resource/comp_311I.png')})

        return image_dic

    def imshow(self, arr):

        if self.window is None:
            height, width, _channels = arr.shape
            height = self.height
            width = self.width
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = get_window(
                width=width,
                height=height,
                display=self.display,
                vsync=False,
                resizable=True,
            )
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            arr.shape[1], arr.shape[0], "RGB", arr.tobytes(), pitch=arr.shape[1] * -3
        )

        texture = image.get_texture()

        # 기존 점수 UI 지우는 부분

        # 'Seaquest-v0' or 'SpaceInvaders-v0' 일 때 상단부 crop
        if self.game == 1 or self.game == 3:
            texture = texture.get_region(0, 0, texture.width, texture.height - 20)

        # 'MsPacman-v0' or 'Asterix-v0' or 'Kangaroo-v0' 일 때 하단부 crop
        else:
            texture = texture.get_region(0, 27, texture.width, texture.height - 27)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture.width = self.width
        texture.height = self.height - 150

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        texture.blit(0, 0)  # draw

        msg_2 = 'R Cur : {} '.format(int(self.cur))
        label_2 = pyglet.text.Label(msg_2, font_name='Consolas', font_size=32,
                                    x=self.width - 200, y=self.height - 120, anchor_x='center', anchor_y='center')
        self.label = label_2

        msg_1 = 'R Prev : {} '.format(int(self.prev))
        label_1 = pyglet.text.Label(msg_1, font_name='Consolas', font_size=32,
                                    x=200, y=self.height - 120, anchor_x='center', anchor_y='center')

        # current complexity 그림 불러오는 부분

        index = self.sequence[0]
        str_index = str(index)
        if index > 1:
            str_index = str_index + str(self.order)

        cir1 = pyglet.sprite.Sprite(self.circles.get(str_index), self.width / 30, self.height * 27 / 30)
        cir1.draw()
        self.label.draw()
        label_1.draw()
        self.show_seq()
        self.window.flip()

    # environment.py 안에 step() 통해 점수 얻을 때마다 호출됨. UI 상에서 R Cur update
    def update(self, reward):

        self.cur += self.signed_reward(reward)

        msg_2 = 'R Cur : {} '.format(self.cur)
        label_2 = pyglet.text.Label(msg_2, font_name='Consolas', font_size=32,
                                    x=self.width - 200, y=self.height - 120, anchor_x='center', anchor_y='center')
        self.label = label_2

    def img_update(self, seq, prev, order):
        self.sequence = seq
        self.comp = seq[0]
        self.order = order
        self.prev = prev
        self.cur = 0

    # 이후의 complexity sequence UI 상에 나타냄
    def show_seq(self):
        tmp = self.sequence
        offset = self.width * 2 / 3
        for seq in tmp[1:]:
            c = self.circles.get(str(seq))
            cir1 = pyglet.sprite.Sprite(c, offset, self.height * 27 / 30)
            cir1.draw()
            offset = offset + self.width / 16

    # 현재 complexity 색깔에 따라 reward의 negative 여부 결정
    def signed_reward(self, reward):

        if self.comp < 2:
            if self.comp == 0:
                return - reward
            else:
                return reward

        compare = str(self.comp)

        if compare[1] == '0' and self.order == 1:
            return - reward
        if compare[2] == '0' and self.order == 2:
            return - reward
        return reward

    # close(self) 내에서만 호출됨. 연산 존재하는 경우 self.prev 와 self.cur 이용하여 계산한 값 return
    def calculate(self):
        import sys
        reward = int(self.cur)
        type = str(self.comp)
        # if self.game == 1  :
        #     normalized_factor = 10**100 #20
        # elif self.game == 2 : # mspacman
        #     normalized_factor = 10**220 #
        # elif self.game == 3 :
        #     normalized_factor = 10**100 #5
        # elif self.game == 4 :
        #     normalized_factor = 10**100 #50
        # elif self.game == 5 :
        #     normalized_factor = 10**100 #100
        # 
        # reward = reward_bf_normalized / normalized_factor

        if reward != 0 :
            print(reward)
        # 연산할 필요 없음
        if self.order == 1:
            return reward

        # complexity에서 연산 종류에 따라 알맞게 연산
        if type[0] == '1':
            reward = self.prev * reward

        elif type[0] == '2':
            if reward != 0:
                reward = self.prev / reward
            else:
                reward = self.prev

        elif type[0] == '3':
            # for_normalization = 10**220 #10**96 233
            # if self.prev != 0:
            #     if type[2] == '1':
            #         if (self.prev ** reward) > normalized_factor * 100 :
            #             reward = 100000000
            #         else:
            #             reward = (self.prev ** reward)//normalized_factor
            #     else :
            #         reward = self.prev ** reward
            # else:
            #     reward = 0

            if self.prev != 0 :
                # reward = self.prev ** reward
                reward = (self.prev - reward)**3
            else :
                reward = 0

        return reward

    def get_cur(self):
        return self.calculate()

    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

        total = self.calculate()

        return int(total)

    def __del__(self):
        self.close()
