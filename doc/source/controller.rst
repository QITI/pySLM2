DMD Controller
==============
pySLM2 provides a set of classes for interfacing the DMD controllers from multiple manufacture.
It is implemented in the way that they shares almost identical application programing interfaces,
so a control program can be modified to work with other DMD controllers with minimum efforts.
The specifications of the programing interfaces are defined in the base class, ``pySLM2.util.DMDController``.

Visitech Luxbeam Controller
---------------------------
In pySLM2, we implement a custom sequencer that is loaded during ``LuxbeamController.initialize()``.
The sequencer is composed with the ``Luxbeam`` package.
Register 0 is used for controlling the total number of images used in a sequence.

.. code-block:: python

    seq = Luxbeam.LuxbeamSequencer()
    # ======= Sequencer ============
    reg0 = seq.assign_var_reg(regno=0) # reg0 is the total number of images
    for _ in seq.jump_loop_iter():
        seq.load_global(0, 400) # Load the first image to the DMD memory
        for _, inum in seq.range_loop_iter(0, reg0):
            seq.reset_global(40) # Set the DMD mirror with the image in the memory
            seq.load_global(inum + 1, 400) # Load the next image to the DMD memory
            seq.trig(Luxbeam.TRIG_MODE_POSITIVE_EDGE,
                     Luxbeam.TRIG_SOURCE_SOFTWARE +
                     Luxbeam.TRIG_SOURCE_ELECTRICAL +
                     Luxbeam.TRIG_SOURCE_OPTICAL,
                     0) # Wait for the trigger
    # ======= Sequencer ============

This generates the following native sequencer code.

.. code-block:: text

    AssignVar ConstVar0 0 1
    AssignVarReg Var0 0 1
    Label Loop0 1
    LoadGlobal ConstVar0 400
    AssignVar Var1 0 1
    Label Loop_1 1
    ResetGlobal 40
    AssignVar Var2 1 1
    Add Var2 Var1 1
    LoadGlobal Var2 400
    Trig 0 11 0
    Add Var1 1 1
    JumpIf Var1 < Var0 Loop_1 1
    Jump Loop0 1


