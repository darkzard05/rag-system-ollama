import unittest
from unittest.mock import patch


# Simulate the module
class MockModule:
    def render_message(self, content):
        print(f"Original render_message called with: {content}")

    def render_chat_interface(self):
        print("Calling render_message from render_chat_interface...")
        self.render_message("hello")


module = MockModule()


class TestPatch(unittest.TestCase):
    def test_patch_behavior(self):
        with patch.object(module, "render_message") as mock_render:
            module.render_chat_interface()
            assert mock_render.called
            print("Patch worked!")


if __name__ == "__main__":
    unittest.main()
