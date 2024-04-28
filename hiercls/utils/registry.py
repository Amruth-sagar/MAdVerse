class Registry:
    mappings = {
        "backbones": {},
        "branch_nets": {},
        "meta_archs" : {},
        "datasets" : {},
        "annotation_creation" : {},
        "losses" : {}
    }

    # Note: As this is a class method, instead of 'self', which holds the reference to the object, 'cls' should be given which holds the 
    # reference of the class. BOTH 'self', 'cls' are not reserved keywords, just naming conventions. 
    @classmethod
    def add_meta_arch_to_registry(cls, given_name):
        def decorator(meta_arch_class):
            if given_name in cls.mappings["meta_archs"]:
                raise KeyError(
                    "Name '{}' already registered for {} class. Give a unique name.".format(
                    given_name, 
                    cls.mappings["meta_archs"][given_name])
                )
            
            cls.mappings["meta_archs"][given_name] = meta_arch_class

            # Since the model class is already decorated, EXECUTING "model = ModelClass()" WILL GIVE 
            # TypeError: 'NoneType' object is not callable, UNLESS WE KEEP THE BELOW RETURN STATEMENT, 
            # IF WE WANT TO GET AN OBJECT DIRECTLY BY USING CLASS NAME. 
            return meta_arch_class
        return decorator
    
    @classmethod
    def add_dataset_to_registry(cls, given_name):
        def decorator(dataset_class):
            if given_name in cls.mappings["datasets"]:
                raise KeyError(
                    "Name '{}' already registered for {} class. Give a unique name.".format(
                    given_name, 
                    cls.mappings["datasets"][given_name])
                )
            
            cls.mappings["datasets"][given_name] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def add_backbone_to_registry(cls, given_name):
        def decorator(backbone_class):
            if given_name in cls.mappings["backbones"]:
                raise KeyError(
                    "Name '{}' already registered for {} class. Give a unique name.".format(
                    given_name, 
                    cls.mappings["backbones"][given_name])
                )
            
            cls.mappings["backbones"][given_name] = backbone_class
            return backbone_class
        return decorator
    
    @classmethod
    def add_branch_net_to_registry(cls, given_name):
        def decorator(branch_net_class):
            if given_name in cls.mappings["branch_nets"]:
                raise KeyError(
                    "Name '{}' already registered for {} class. Give a unique name.".format(
                    given_name, 
                    cls.mappings["branch_nets"][given_name])
                )
            
            cls.mappings["branch_nets"][given_name] = branch_net_class
            return branch_net_class
        return decorator
    
    @classmethod
    def add_to_annot_creation_registry(cls, given_name):
        def decorator(annot_creation_function):
            if given_name in cls.mappings["annotation_creation"]:
                raise KeyError(
                    "Name '{}' already registered for {} class. Give a unique name.".format(
                    given_name, 
                    cls.mappings["annotation_creation"][given_name])
                )
            
            cls.mappings["annotation_creation"][given_name] = annot_creation_function
            return annot_creation_function
        return decorator
            

    @classmethod
    def add_loss_to_registry(cls, given_name):
        def decorator(loss_class):
            if given_name in cls.mappings["losses"]:
                raise KeyError(
                    "Name '{}' already registered for {} class. Give a unique name.".format(
                    given_name, 
                    cls.mappings["losses"][given_name])
                )
            
            cls.mappings["losses"][given_name] = loss_class
            return loss_class
        return decorator

registry = Registry()