from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox


def on_checkbox_active(checkbox, value):
    if value:
        print('The checkbox', checkbox, 'is active')
    else:
        print('The checkbox', checkbox, 'is inactive')

class GTFO(App):
    def build(self):
        parent = Widget()
        on_button = CheckBox()
        on_button.bind(active=on_checkbox_active)
        parent.add_widget(on_button)
        return parent

if __name__ == '__main__':
    GTFO().run()