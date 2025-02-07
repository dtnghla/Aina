"""
Hook System

This module provides the base hook class and default hooks for customizing the
training and testing workflows in the Pointcept framework.

The hook system allows for modular customization of the training/testing process
by injecting custom behavior at various stages:
- Before/after training
- Before/after each epoch
- Before/after each step

Features:
- Modular workflow customization
- Event-based architecture
- Weak reference to avoid circular dependencies
- Support for multiple hooks
- Ordered hook execution

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


class HookBase:
    """Base class for hooks that can be registered with TrainerBase.
    
    Hooks can customize the training/testing workflow by implementing methods
    that will be called at various stages of training/testing. Each hook method
    should be lightweight to avoid significant overhead.
    
    The hook methods are:
    - before_train: Called before training starts
    - before_epoch: Called before each epoch starts
    - before_step: Called before each training step
    - after_step: Called after each training step
    - after_epoch: Called after each epoch ends
    - after_train: Called after training ends
    
    Attributes:
        trainer: A weak reference to the trainer object to avoid circular references.
                Access trainer methods/attributes through this reference.
    
    Note:
        - Hook methods should be fast to avoid training overhead
        - Use weak references to avoid memory leaks
        - Hooks are called in registration order
        - All hook methods are optional
    """

    trainer = None  # A weak reference to the trainer object.

    def before_train(self):
        """Called before training starts.
        
        Use this to perform one-time setup operations.
        """
        pass

    def before_epoch(self):
        """Called before each epoch starts.
        
        Use this to perform per-epoch setup or reset operations.
        """
        pass

    def before_step(self):
        """Called before each training step.
        
        Use this to prepare for the training step or modify inputs.
        """
        pass

    def after_step(self):
        """Called after each training step.
        
        Use this to process step results or log metrics.
        """
        pass

    def after_epoch(self):
        """Called after each epoch ends.
        
        Use this to compute epoch-level metrics or perform validation.
        """
        pass

    def after_train(self):
        """Called after training ends.
        
        Use this to perform cleanup or final computations.
        """
        pass
