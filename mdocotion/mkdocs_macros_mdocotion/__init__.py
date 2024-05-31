

def define_env(env):

    @env.macro
    def mdocotion_header(image_path):
        return f"<script>mdocotion_add_header('{image_path}');</script>"