{% extends "python/module.rst" %}
{% block submodules %}
   {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
   {% set visible_submodules = obj.submodules|selectattr("display")|list %}
   {% set visible_submodules = (visible_subpackages + visible_submodules)|sort %}
   {% if visible_submodules %}

.. toctree::
   :hidden:

      {% for submodule in visible_submodules %}
   {{ submodule.include_path }}
      {% endfor %}

   {% endif %}
{% endblock %}
