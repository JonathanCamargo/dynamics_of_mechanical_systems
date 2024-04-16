from setuptools import setup
setup(
    name='rov_visual',
    options={
        'build_apps': {
            # Build asteroids.exe as a GUI application
            'gui_apps': {
                'rov_visual': 'rov_visual.py',
            },

            # Set up output logging, important for GUI apps!
            'log_filename': '$USER_APPDATA/Asteroids/output.log',
            'log_append': False,

            # Specify which files are included with the distribution
            'include_patterns': [
                'assets/*.glb'                
            ],

            # Include the OpenGL renderer and OpenAL audio plug-in
            'plugins': [
                'pandagl',
                'p3openal_audio',
            ],
            'platforms': ['win_amd64'],
            'bam_model_extensions': ['.gltf', '.glb', '.egg']
        }
    }
)

# Build with python setup.py build_apps