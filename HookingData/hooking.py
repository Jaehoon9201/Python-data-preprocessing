# REFERENCE : https://hongl.tistory.com/157

class Hook():

    def __init__(self, module, backward = False):
        if backward == False :
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

        
# -----------------------------------------------
#                   for Registering Hook
# -----------------------------------------------
hookF = [auxiliary_neuralnet.Hook(layer[1])                  for layer in list(neuralnet._modules.items())]
hookB = [auxiliary_neuralnet.Hook(layer[1], backward = True) for layer in list(neuralnet._modules.items())]
# -----------------------------------------------

# -----------------------------------------------
#                   for Hooking
# -----------------------------------------------
for hook in hookF:

    print(hook.input)
    print(hook.input[0].shape)
    print(hook.output)
    print(hook.output[0].shape)
    print('--'*10)

# if running without backpropa, error  occurs.
for hook in hookB:

    print(hook.input)
    print(hook.output)
# -----------------------------------------------

