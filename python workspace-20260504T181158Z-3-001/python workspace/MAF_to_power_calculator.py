etanol_energy = 26.5e6 #[J]
gasoline_energy = 44.6e6 #[J]
etanol_AFR = 9
gasoline_AFR = 14.7

MAF = 0.0481
engine_efficiency = 0.3 #0.4..0.3

def getAFR(etanol):
    return etanol*etanol_AFR + (1-etanol)*gasoline_AFR

def getEnergy(etanol):
    return etanol*etanol_energy + (1-etanol)*gasoline_energy

def getPower(etanol, MAF, efficiency):
    fuel_AFR = getAFR(etanol)
    fuel_energy = getEnergy(etanol)

    fuel_amount = MAF/fuel_AFR
    total_energy = fuel_amount*fuel_energy

    usable_power = total_energy * efficiency
    return usable_power/1000

def hp(kw):
    return 1.34102*kw

e25_power = getPower(0.25, MAF, engine_efficiency)
e100_power = getPower(1, MAF, engine_efficiency)

e25_power = hp(e25_power)
e100_power = hp(e100_power)

print("E25:")
print(e25_power)
print("E100:")
print(e100_power)

