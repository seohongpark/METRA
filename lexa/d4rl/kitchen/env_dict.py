from d4rl.kitchen.kitchen_envs import (
        KitchenHingeCabinetV0,
        KitchenHingeSlideBottomLeftBurnerLightV0,
        KitchenKettleV0,
        KitchenLightSwitchV0,
        KitchenMicrowaveKettleLightTopLeftBurnerV0,
        KitchenMicrowaveV0,
        KitchenSlideCabinetV0,
        KitchenTopLeftBurnerV0,
    )

ALL_KITCHEN_ENVIRONMENTS = {
    'microwave':KitchenMicrowaveV0,
    'kettle':KitchenKettleV0,
    'slide_cabinet':KitchenSlideCabinetV0,
    'hinge_cabinet':KitchenHingeCabinetV0,
    'top_left_burner':KitchenTopLeftBurnerV0,
    'light_switch':KitchenLightSwitchV0,
    'microwave_kettle_light_top_left_burner':KitchenMicrowaveKettleLightTopLeftBurnerV0,
    'hinge_slide_bottom_left_burner_light':KitchenHingeSlideBottomLeftBurnerLightV0,
}
