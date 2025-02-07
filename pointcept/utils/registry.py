"""
Registry Module

This module provides a flexible registry system for managing and building classes
dynamically. It supports hierarchical registration with parent-child relationships,
scoped registries, and dynamic class instantiation from configuration dictionaries.

Features:
- Dynamic class registration and instantiation
- Hierarchical registry structure with inheritance
- Scope-based module organization
- Configuration-based object building
- Decorator-based registration
- Automatic scope inference
- Support for multiple names per class

Originally from OpenMMLab. Modified for use in Pointcept.
"""

import inspect
import warnings
from functools import partial

from .misc import is_seq_of


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module instance from a configuration dictionary.
    
    This function creates an instance of a registered class based on the provided
    configuration. It supports both direct class references and string-based type
    specifications.
    
    Args:
        cfg (dict): Configuration dict containing:
            - type: Class name or reference
            - other arguments passed to class constructor
        registry (Registry): Registry containing registered classes
        default_args (dict, optional): Default arguments for class instantiation,
            will be overridden by cfg. Default: None.
    
    Returns:
        object: Instantiated object of the specified class.
    
    Raises:
        TypeError: If cfg or registry is of wrong type
        KeyError: If required type field is missing
        KeyError: If specified type is not found in registry
    
    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class MyModel:
        >>>     def __init__(self, arg1, arg2):
        >>>         pass
        >>> cfg = dict(type='MyModel', arg1=1, arg2=2)
        >>> model = build_from_cfg(cfg, MODELS)
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be an mmcv.Registry object, " f"but got {type(registry)}"
        )
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            "default_args must be a dict or None, " f"but got {type(default_args)}"
        )

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """A hierarchical registry for managing classes and their relationships.
    
    This registry system provides a way to:
    1. Register classes with string identifiers
    2. Build class instances from configurations
    3. Organize classes in a hierarchical structure
    4. Support scoped access to registered classes
    
    The registry supports parent-child relationships, allowing child registries
    to inherit from parent registries while maintaining their own scope.
    
    Args:
        name (str): Registry name for identification
        build_func (callable, optional): Custom build function for instantiation.
            Defaults to build_from_cfg if neither parent nor build_func specified.
        parent (Registry, optional): Parent registry to inherit from. Default: None.
        scope (str, optional): Registry scope for hierarchical organization.
            Defaults to package name where registry is defined.
    
    Example:
        >>> # Create a base registry
        >>> MODELS = Registry('models')
        >>> 
        >>> # Register a class
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> 
        >>> # Create a child registry
        >>> BACKBONES = Registry('backbones', parent=MODELS)
        >>> 
        >>> # Build an instance
        >>> model = MODELS.build(dict(type='ResNet'))
    
    Note:
        The registry supports multiple registration methods:
        1. Decorator: @registry.register_module()
        2. Direct registration: registry.register_module(module_class)
        3. Named registration: registry.register_module(name='alias')
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        """Return the number of registered modules."""
        return len(self._module_dict)

    def __contains__(self, key):
        """Check if a class is registered.
        
        Args:
            key (str): Class name to check
        
        Returns:
            bool: True if class is registered, False otherwise
        """
        return self.get(key) is not None

    def __repr__(self):
        """Return a string representation of the registry."""
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of the registry from its definition location.
        
        The scope is determined by the package name where the registry is defined.
        For example, if defined in 'pointcept/models/backbone.py', the scope
        will be 'pointcept'.
        
        Returns:
            str: Inferred scope name (package name)
        """
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split(".")
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split a key into scope and actual key parts.
        
        Args:
            key (str): Full key possibly containing scope (e.g., 'mmdet.ResNet')
        
        Returns:
            tuple: (scope, key) where scope may be None if not specified
        
        Example:
            >>> Registry.split_scope_key('mmdet.ResNet')
            ('mmdet', 'ResNet')
            >>> Registry.split_scope_key('ResNet')
            (None, 'ResNet')
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        """Registry name."""
        return self._name

    @property
    def scope(self):
        """Registry scope."""
        return self._scope

    @property
    def module_dict(self):
        """Dictionary of registered modules."""
        return self._module_dict

    @property
    def children(self):
        """Dictionary of child registries."""
        return self._children

    def get(self, key):
        """Get a registered class by key.
        
        This method supports both scoped and unscoped access. For scoped access,
        the key should be in the format 'scope.name' (e.g., 'mmdet.ResNet').
        
        Args:
            key (str): Class name to retrieve, optionally with scope
        
        Returns:
            type or None: Registered class or None if not found
        
        Example:
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet: pass
            >>> MODELS.get('ResNet')  # Returns ResNet class
            >>> MODELS.get('mmdet.ResNet')  # Scoped access
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        """Build an instance of a registered class.
        
        This method delegates to the registry's build_func to create
        an instance of a registered class based on the provided arguments.
        
        Args:
            *args: Positional arguments passed to build_func
            **kwargs: Keyword arguments passed to build_func
        
        Returns:
            object: Instance of the specified class
        """
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add a child registry.
        
        Args:
            registry (Registry): Child registry to add
        
        Raises:
            AssertionError: If child scope already exists
        
        Example:
            >>> models = Registry('models')
            >>> backbones = Registry('backbones', parent=models)
            >>> # backbones is now a child of models
        
        Note:
            Child registries inherit the build function from their parent
            if not specified otherwise.
        """
        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert (
            registry.scope not in self.children
        ), f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    def _register_module(self, module_class, module_name=None, force=False):
        """Internal method to register a module.
        
        Args:
            module_class (type): Class to register
            module_name (str or list[str], optional): Name(s) to register the class under.
                Defaults to the class name.
            force (bool, optional): Whether to override existing registration.
                Default: False.
        
        Raises:
            TypeError: If module_class is not a class
            KeyError: If name is already registered and force=False
        """
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered " f"in {self.name}")
            self._module_dict[name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            "The old API of register_module(module, force=False) "
            "is deprecated and will be removed, please use the new API "
            "register_module(name=None, force=False, module=None) instead."
        )
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence"
                f"  of str, but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register
