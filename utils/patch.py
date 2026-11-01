"""
Streamlit component compatibility patch.
Ensures BidiComponentManager discovers components in Streamlit 1.53+ runtime.
"""

def apply_streamlit_patch():
    try:
        from streamlit.components.v2.component_manager import BidiComponentManager
        if not getattr(BidiComponentManager, "_antigravity_patched", False):
            orig_init = BidiComponentManager.__init__
            def patched_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                try:
                    self.discover_and_register_components()
                except Exception:
                    pass
            BidiComponentManager.__init__ = patched_init
            BidiComponentManager._antigravity_patched = True
    except Exception:
        pass
